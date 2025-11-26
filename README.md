# Vision-Language Model Chatbot

A custom Vision-Language Model (VLM) that combines CLIP vision encoder with Vicuna-7B language model for image understanding and visual question answering. Built from scratch with efficient training pipeline and deployment-ready inference.

## Overview

**Problem**: Large vision-language models like GPT-4V and Gemini are powerful but expensive and closed-source. Building a custom VLM requires aligning vision and language modalities efficiently.

**Solution**: This project implements a modular VLM architecture that:
- Leverages pre-trained CLIP (vision) and Vicuna-7B (language) models
- Trains only a lightweight projection layer (4.2M parameters) to align modalities
- Uses 4-bit quantization for memory-efficient deployment (~8GB VRAM)
- Implements production-ready training with mixed precision, gradient accumulation, and early stopping

## Key Technical Achievements

- **Efficient Training Pipeline**: Custom PyTorch training loop with AMP, gradient accumulation (32x effective batch size), and cosine LR scheduling
- **Memory Optimization**: 4-bit quantized LLM reduces VRAM from 28GB to 8GB while maintaining quality
- **Robust Training**: Gradient clipping, NaN detection, and checkpoint resumption for stable training
- **End-to-End Implementation**: Built complete pipeline from data downloading to web deployment
- **Validated Results**: Achieved 35% train loss reduction and 12% validation loss reduction in just 3 epochs on COCO dataset

## Architecture

The model follows a three-component architecture inspired by LLaVA and MiniGPT-4:

```
Image Input (224x224)
    ↓
CLIP Vision Encoder (ViT-L/14)
    ↓ [1024-dim embeddings]
Projection Layer (Learnable)
    ↓ [4096-dim embeddings]
Vicuna-7B Language Model
    ↓
Text Output
```

### Components

1. **Vision Encoder** (CLIP ViT-L/14 - Frozen)
   - Pre-trained on 400M image-text pairs
   - Extracts rich visual features: 1024-dimensional embeddings
   - Frozen during training to preserve learned representations

2. **Projection Layer** (Trainable - 4.2M params)
   - Linear transformation: 1024 → 4096 dimensions
   - Maps vision features to language model embedding space
   - Only 0.11% of total model parameters
   - Xavier normal initialization (gain=0.02) for stable training

3. **Language Model** (Vicuna-7B-v1.5 - Frozen)
   - 7B parameter instruction-tuned LLM
   - 4-bit quantization (bitsandbytes) for efficient inference
   - Frozen during training to leverage pre-trained language understanding

**Key Insight**: Only the projection layer is trained, making the approach highly parameter-efficient compared to full fine-tuning (4.2M vs 7B parameters).

## Features

- ✅ Image captioning and visual question answering
- ✅ Memory-efficient inference with 4-bit quantization (~8GB VRAM)
- ✅ Production-ready training pipeline with AMP and gradient accumulation
- ✅ Automated COCO dataset downloading and preprocessing
- ✅ Interactive web demo with Gradio
- ✅ Comprehensive error handling (NaN detection, gradient clipping)
- ✅ Training resumption from checkpoints
- ✅ Real-time validation with generated caption examples

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd vlm-chatbot

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ RAM

### Basic Usage

```python
from models.vlm import VLMChatbot

# Initialize model with 4-bit quantization
model = VLMChatbot(load_in_4bit=True)

# Load trained projection weights
model.load_checkpoint("checkpoints/vlm_projection_best_improved.pth")

# Generate caption or answer questions
response = model.chat(
    image_path="path/to/image.jpg",
    question="Describe this image.",
    max_new_tokens=100
)
print(response)
```

### Web Interface

Launch interactive Gradio demo:

```bash
python inference/gradio.py
```

Access at `http://localhost:7860` for visual question answering with any image.

## Training

### Training Configuration

```bash
python training/train_projection_improved.py
```

**Hyperparameters**:
```yaml
Batch size: 2 (per GPU)
Gradient accumulation: 16 steps (effective batch size: 32)
Learning rate: 2e-4
LR schedule: Cosine decay with 200-step warmup
Optimizer: AdamW (weight_decay=0.01)
Epochs: 20 (with early stopping patience=3)
Mixed precision: FP16 (AMP)
Gradient clipping: max_norm=1.0
```

### Training Pipeline Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Mixed Precision (AMP)** | FP16 computation with FP32 master weights | 2x faster training, 50% less memory |
| **Gradient Accumulation** | Accumulate gradients over 16 steps | Large effective batch size on limited VRAM |
| **Cosine LR Schedule** | Warmup + cosine decay to 0.1x initial LR | Better convergence, prevents overfitting |
| **Early Stopping** | Patience=3 epochs on validation loss | Automatic training termination |
| **NaN Detection** | Skip batches with NaN/Inf gradients | Robust training with quantized models |
| **Checkpoint Resume** | Save/load training state | Resume from interruptions |
| **Generative Validation** | Show actual generated captions | Qualitative assessment during training |

### Training Results (3 Epochs on COCO)

| Epoch | Train Loss | Val Loss | Learning Rate | Improvement |
|-------|-----------|----------|---------------|-------------|
| 1 | 2.471 | 1.873 | 0.000145 | Baseline |
| 2 | 1.792 | 1.705 | 0.000199 | ↓27.5% train, ↓9.0% val |
| 3 | 1.611 | 1.650 | 0.000196 | ↓34.8% train, ↓11.9% val |

**Key Observations**:
- Consistent loss reduction on both train and validation sets (no overfitting)
- Learning rate successfully warming up (0.000145 → 0.000199)
- Model learns basic image description capabilities after just 3 epochs
- Recommended: 10-15 epochs for production-quality outputs

### Dataset

Training uses **COCO 2017 Validation** dataset:
- 5,000 images with human-annotated captions
- Automatic download on first run (~1GB images + 20MB annotations)
- Average 5 captions per image

```bash
# Manual dataset download (optional)
python training/data_downloader.py
```

## Technical Implementation Details

### Memory Optimization

**Challenge**: Vicuna-7B requires ~28GB VRAM in FP16, exceeding most consumer GPUs.

**Solution**: 4-bit quantization with bitsandbytes
```python
# Before: 28GB VRAM (FP16)
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")

# After: 8GB VRAM (4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    load_in_4bit=True,
    device_map="auto"
)
```

**Impact**: 3.5x memory reduction with minimal quality loss

### Gradient Accumulation Strategy

**Challenge**: Batch size of 2 is too small for stable training.

**Solution**: Accumulate gradients over 16 steps
```python
for i, batch in enumerate(dataloader):
    loss = forward_pass(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Impact**: Effective batch size of 32 on 8GB VRAM

### Training Stability

Key techniques for stable training with quantized models:

1. **Gradient Clipping**: `clip_grad_norm_(parameters, max_norm=1.0)`
2. **NaN Detection**: Skip batches with numerical issues
3. **Careful Initialization**: Xavier normal with small gain (0.02)
4. **Mixed Precision**: GradScaler handles overflow/underflow

## Performance Metrics

| Metric | Value | Hardware |
|--------|-------|----------|
| Inference Speed | ~2-3 seconds/image | NVIDIA RTX 3080 |
| Training Speed | ~1.8 seconds/batch | NVIDIA RTX 3080 |
| Memory Usage | 8GB VRAM | 4-bit quantization |
| Model Size | 4.2M trainable params | 0.11% of total |
| Checkpoint Size | ~17MB | Projection weights only |

## Project Structure

```
vlm-chatbot/
├── models/                          # Model implementations
│   ├── vision_encoder.py           # CLIP ViT-L/14 wrapper
│   ├── projection_layer.py         # Basic projection layer
│   ├── projection_layer_improved.py # Improved initialization
│   ├── language_model.py           # Vicuna-7B wrapper with 4-bit
│   └── vlm.py                      # Main VLM model
├── training/                        # Training scripts
│   ├── train_projection_improved.py # Full training pipeline
│   └── data_downloader.py          # COCO dataset utilities
├── inference/                       # Inference and deployment
│   └── gradio.py                   # Interactive web demo
├── optimization/                    # Performance tools
│   ├── benchmarking.py             # Timing and profiling
│   └── caching.py                  # Response caching
├── tests/                          # Unit tests
├── checkpoints/                    # Model checkpoints
│   ├── vlm_projection_best_improved.pth
│   └── training_history_improved.json
└── data/                           # Datasets
    └── coco_val/                   # COCO validation set
```

## Current Limitations & Future Work

### Known Limitations

**Early Training Stage** (3-5 epochs):
- HTML/XML artifacts in generated text (`</h>`, `</b>`)
- Occasional repetitive phrases
- Limited instruction-following (general description vs. specific questions)

**Recommended**: Train for 10-15 epochs for production quality

### Future Enhancements

- [ ] Implement BLEU, ROUGE, METEOR metrics
- [ ] LoRA fine-tuning for language model (parameter-efficient)
- [ ] Multi-layer projection network (MLP vs. single linear)
- [ ] Support alternative vision encoders (DINOv2, SigLIP)
- [ ] Beam search and nucleus sampling for generation
- [ ] Batch inference support
- [ ] Distributed training across multiple GPUs

**Completed**:
- [x] Mixed precision training (AMP)
- [x] Cosine LR scheduling with warmup
- [x] Early stopping
- [x] Gradient accumulation
- [x] 4-bit quantization

## Technical Stack

- **Framework**: PyTorch 2.0+
- **Vision**: CLIP (OpenAI)
- **Language**: Vicuna-7B (LMSYS)
- **Quantization**: bitsandbytes (4-bit)
- **Optimization**: torch.optim.AdamW, AMP
- **Data**: COCO 2017 dataset
- **UI**: Gradio

## References & Acknowledgments

This project builds upon:

1. **LLaVA** (Liu et al., 2023): Visual instruction tuning
   - Paper: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

2. **MiniGPT-4** (Zhu et al., 2023): Vision-language alignment
   - Paper: [MiniGPT-4: Enhancing Vision-Language Understanding](https://arxiv.org/abs/2304.10592)

3. **CLIP** (Radford et al., 2021): Vision encoder
   - Repository: [OpenAI CLIP](https://github.com/openai/CLIP)

4. **Vicuna** (Chiang et al., 2023): Language model
   - Model: [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)

5. **COCO Dataset** (Lin et al., 2014): Training data
   - Website: [Microsoft COCO](https://cocodataset.org/)

## License

MIT License - see LICENSE file for details.

## Author

Built as a deep learning research project exploring efficient vision-language model alignment.

---

**Note**: This is a research/educational project. For production use, consider additional evaluation, safety filtering, and bias mitigation.
