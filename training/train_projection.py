"""
Training script for VLM projection layer.
Freezes vision encoder and language model, trains only the projection layer
"""

import json
import math
import os
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.vlm import VLMChatbot
from training.data_downloader import COCODownloader, SimpleCOCODataset


def freeze_model_except_projection(model):
    """Freeze all parameters except projection layer."""
    # Freeze vision encoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False

    # Freeze language model
    for param in model.language_model.parameters():
        param.requires_grad = False

    # Unfreeze projection layer
    for param in model.projection.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Cosine scheduler for learning rate - between the initial lr to min_lr_ratio * initial_lr
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    model.vision_encoder.eval()
    model.language_model.eval()
    model.projection.train()

    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for batch_idx, batch in enumerate(progress_bar):
        try:
            images = batch["image"]
            questions = batch["question"]
            answers = batch["answer"]

            # Project vision features
            with torch.no_grad():
                vision_features = model.vision_encoder(images)
            vision_embeddings = model.projection(vision_features)

            # Forward pass
            with autocast("cuda", enabled=config.get("use_amp", True)):
                losses = []
                for i, (question, answer) in enumerate(zip(questions, answers)):
                    single_vision_emb = vision_embeddings[i : i + 1]
                    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
                    text_inputs = model.language_model.tokenizer(
                        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
                    ).to(device)

                    # Forward through model
                    combined_embeds, combined_mask = model.language_model.prepare_inputs_with_vision(
                        text_inputs, single_vision_emb
                    )

                    num_vision_tokens = single_vision_emb.shape[1]

                    prompt_without_answer = f"USER: <image>\n{question}\nASSISTANT:"
                    prompt_tokens = model.language_model.tokenizer(prompt_without_answer, add_special_tokens=False)[
                        "input_ids"
                    ]
                    answer_start_idx = len(prompt_tokens)

                    # Create labels
                    vision_labels = torch.full((1, num_vision_tokens), -100, dtype=torch.long, device=device)
                    text_labels = text_inputs["input_ids"].clone()
                    text_labels[0, :answer_start_idx] = -100
                    labels = torch.cat([vision_labels, text_labels], dim=1)
                    outputs = model.language_model(
                        inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels
                    )
                    losses.append(outputs.loss / config.get("gradient_accumulation_steps", 1))

            # Average loss for batch
            if losses:
                loss = torch.stack(losses).mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nSkipping {batch_idx}, NaN/Inf loss detected")
                    continue

                # Backward pass
                scaler.scale(loss).backward()

                if (batch_idx + 1) % config.get("gradient_accumulation_steps", 1) == 0:
                    scaler.unscale_(optimizer)

                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.projection.parameters(), max_norm=config["max_grad_norm"]
                    )

                    # Check for NaN gradients
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\nWarning: NaN/Inf gradients detected in batch {batch_idx}, skipping...")
                        optimizer.zero_grad()
                        scaler.update()
                        continue

                    # Optimizer step with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Clear CUDA cache
                    if torch.cuda.is_available() and batch_idx % 50 == 0:
                        torch.cuda.empty_cache()

                    current_lr = scheduler.get_last_lr()[0]
                else:
                    grad_norm = 0.0
                    current_lr = scheduler.get_last_lr()[0]

                total_loss += loss.item() * config.get("gradient_accumulation_steps", 1)
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * config.get('gradient_accumulation_steps', 1):.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate_generative(model, dataloader, device, epoch, config):
    """
    Generative validation with autoregressive generation.
    """
    model.eval()
    model.vision_encoder.eval()
    model.language_model.eval()
    model.projection.eval()

    total_loss = 0
    num_batches = 0

    print(f"\nValidating epoch {epoch} (generative mode)")
    progress_bar = tqdm(dataloader, desc="Validation")

    examples_shown = 0
    max_examples = config.get("val_examples_to_show", 3)
    max_new_tokens = config.get("val_generation_max_tokens", 50)

    for batch_idx, batch in enumerate(progress_bar):
        try:
            images = batch["image"]
            questions = batch["question"]
            answers = batch["answer"]

            vision_features = model.vision_encoder(images)
            vision_embeddings = model.projection(vision_features)

            # Forward pass
            losses = []
            for i, (question, answer) in enumerate(zip(questions, answers)):
                single_vision_emb = vision_embeddings[i : i + 1]

                # Calculate loss with teacher forcing
                prompt_with_answer = f"USER: <image>\n{question}\nASSISTANT: {answer}"
                text_inputs = model.language_model.tokenizer(
                    prompt_with_answer, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(device)

                combined_embeds, combined_mask = model.language_model.prepare_inputs_with_vision(
                    text_inputs, single_vision_emb
                )

                # Create proper labels (only supervise answer)
                num_vision_tokens = single_vision_emb.shape[1]
                prompt_without_answer = f"USER: <image>\n{question}\nASSISTANT:"
                prompt_tokens = model.language_model.tokenizer(prompt_without_answer, add_special_tokens=False)[
                    "input_ids"
                ]
                answer_start_idx = len(prompt_tokens)

                vision_labels = torch.full((1, num_vision_tokens), -100, dtype=torch.long, device=device)
                text_labels = text_inputs["input_ids"].clone()
                text_labels[0, :answer_start_idx] = -100
                labels = torch.cat([vision_labels, text_labels], dim=1)

                outputs = model.language_model(
                    inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels
                )

                losses.append(outputs.loss)

                # Generate autoregressive text
                if examples_shown < max_examples and batch_idx < 3:
                    # Generate without teacher forcing
                    prompt_without_answer = f"USER: <image>\n{question}\nASSISTANT:"
                    gen_text_inputs = model.language_model.tokenizer(prompt_without_answer, return_tensors="pt").to(
                        device
                    )

                    gen_combined_embeds, gen_combined_mask = model.language_model.prepare_inputs_with_vision(
                        gen_text_inputs, single_vision_emb
                    )

                    output_ids = model.language_model.generate(
                        inputs_embeds=gen_combined_embeds,
                        attention_mask=gen_combined_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=False,
                        pad_token_id=model.language_model.tokenizer.pad_token_id,
                        eos_token_id=model.language_model.tokenizer.eos_token_id,
                    )

                    # Decode
                    generated_text = model.language_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    # Extract answer part
                    if "ASSISTANT:" in generated_text:
                        predicted_answer = generated_text.split("ASSISTANT:")[-1].strip()
                    else:
                        predicted_answer = generated_text.strip()

                    print(f"\n--- Validation Example {examples_shown + 1} ---")
                    print(f"Question: {question}")
                    print(f"Target: {answer}")
                    print(f"Generated: {predicted_answer}")
                    examples_shown += 1

            # Average loss
            if losses:
                loss = torch.stack(losses).mean()
                total_loss += loss.item()
                num_batches += 1

        except Exception as e:
            print(f"\nError in validation batch {batch_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    print(f"\n{'='*50}")
    print(f"Validation Results - Epoch {epoch}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"{'='*50}\n")

    return avg_loss


def main():
    # Configuration
    config = {
        "batch_size": 2,  # Small batch due to per-sample LM processing
        "gradient_accumulation_steps": 16,  # Effective batch size = 2 * 16 = 32
        "num_epochs": 20,
        "learning_rate": 2e-4,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "warmup_steps": 200,
        "max_samples": None,  # None = use entire COCO dataset
        "val_split": 0.05,
        "checkpoint_dir": "checkpoints",
        "load_in_4bit": True,
        "use_amp": True,  # Mixed precision training
        "num_workers": 0,
        "early_stopping_patience": 3,  # Early stopping config
        "early_stopping_min_delta": 0.001,
        "resume_from_checkpoint": None,  # Resume config
        "start_epoch": 1,
        "val_generation_max_tokens": 50,  # Validation config
        "val_examples_to_show": 5,
    }

    print("=" * 50)
    print("VLM Projection Layer Training")
    print("=" * 50)
    print(json.dumps(config, indent=2))

    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Initialize model
    print("\nInitializing model")
    model = VLMChatbot(load_in_4bit=config["load_in_4bit"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Freeze model
    freeze_model_except_projection(model)

    # Load checkpoint
    if config.get("resume_from_checkpoint") and os.path.exists(config["resume_from_checkpoint"]):
        print(f"\nLoading checkpoint from {config['resume_from_checkpoint']}")
        model.load_checkpoint(config["resume_from_checkpoint"])
        print("Checkpoint loaded successfully!")
    else:
        if config.get("resume_from_checkpoint"):
            print(f"\nWarning: Checkpoint {config['resume_from_checkpoint']} not found. Starting fresh")

    # Download COCO data
    data_dir = "data/coco_val"
    max_samples = config["max_samples"]

    if not os.path.exists(f"{data_dir}/annotations.json"):
        print("Downloading COCO dataset")
        if max_samples is None:
            print("max_samples is set to None, downloading all samples")
        downloader = COCODownloader(data_dir=data_dir, max_samples=max_samples)
        downloader.prepare_dataset()

    # Load dataset
    full_dataset = SimpleCOCODataset(data_dir=data_dir, max_samples=max_samples)
    if max_samples is None:
        print(f"Using entire COCO dataset: {len(full_dataset)} samples")
    else:
        print(f"Using {len(full_dataset)} samples from COCO dataset")

    # Split into train and validation
    val_size = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])  # type:ignore

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    def collate_fn(x):
        return {
            "image": [item["image"] for item in x],
            "question": [item["question"] for item in x],
            "answer": [item["answer"] for item in x],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01,  # weight decay for regularization
    )

    # Learning rate scheduler with cosine decay
    num_training_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps,
        min_lr_ratio=config["min_lr_ratio"],
    )

    # Gradient scaler for mixed precision training
    scaler = GradScaler("cuda", enabled=config["use_amp"])

    # Training loop
    print("\nStarting training")
    best_val_loss = float("inf")
    training_history = []
    start_epoch = config.get("start_epoch", 1)
    epochs_without_improvement = 0

    # Load training history if resuming
    history_path = os.path.join(config["checkpoint_dir"], "training_history_improved.json")
    if config.get("resume_from_checkpoint") and os.path.exists(history_path):
        print(f"Loading training history from {history_path}")
        with open(history_path, "r") as f:
            training_history = json.load(f)
        if training_history:
            best_val_loss = min(h["val_loss"] for h in training_history)
            print(f"Resuming from epoch {start_epoch} (previous best val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, config)

        # Validation
        val_loss = validate_generative(model, val_loader, device, epoch, config)

        # Track history
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        # Save checkpoint + best model
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"vlm_projection_improved_epoch{epoch}.pth")
        model.save_checkpoint(checkpoint_path)

        if val_loss < best_val_loss - config["early_stopping_min_delta"]:
            best_val_loss = val_loss
            best_path = os.path.join(config["checkpoint_dir"], "vlm_projection_best_improved.pth")
            model.save_checkpoint(best_path)
            print(f"✓ New best model! val_loss={best_val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.4f})")

        # Early stopping
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(f"\nEarly stopping triggered! No improvement for {config['early_stopping_patience']} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

    print("\n" + "=" * 50)
    print("Done")
    print(f"Best model saved to: {os.path.join(config['checkpoint_dir'], 'vlm_projection_best_improved.pth')}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training history saved to: {history_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
