import gradio as gr
import torch

from models.vlm import VLMChatbot

# Load model - set checkpoint_path to your trained model
CHECKPOINT_PATH = None  # Set to "checkpoints/vlm_projection_best.pth" after training

model = VLMChatbot(
    load_in_4bit=True,
    checkpoint_path=CHECKPOINT_PATH
)
model.eval()
print("Model loaded!")


def chat_with_image(image, question, history):
    try:
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)

        with torch.no_grad():
            response = model.chat(temp_path, question, max_new_tokens=150)

        print(f"DEBUG - Question: {question}")
        print(f"DEBUG - Response: '{response}'")
        print(f"DEBUG - Response length: {len(response)}")

        # If response is empty, use a placeholder
        if not response or len(response.strip()) == 0:
            response = "[Model generated empty response - model needs training]"

        # Gradio chatbot expects messages with 'role' and 'content' format
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})
        return history, history

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, history


with gr.Blocks(title="VLM Chatbot") as demo:
    gr.Markdown("# Vision-Language Model Chatbot")
    gr.Markdown("Upload an image and ask questions!")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            question_input = gr.Textbox(label="Ask a question", placeholder="What's in this image?")
            submit_btn = gr.Button("Send", variant="primary")

    state = gr.State([])

    submit_btn.click(chat_with_image, inputs=[image_input, question_input, state], outputs=[chatbot, state])

    question_input.submit(chat_with_image, inputs=[image_input, question_input, state], outputs=[chatbot, state])

    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

    gr.Examples(
        examples=[
            ["data/test_images/dog.jpg", "What breed is this dog?"],
            ["data/test_images/cat.jpg", "Describe this cat."],
        ],
        inputs=[image_input, question_input],
    )

if __name__ == "__main__":
    demo.launch(share=True)
