import gradio as gr

# Define a function that will be called when the send button is clicked
def send_message(message):
    return f"You said: {message}"

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        input_box = gr.Textbox(placeholder="Type a message...", show_label=False)
        send_button = gr.Button("Send")
    
    # Display the response from the function
    output_box = gr.Textbox(show_label=False, interactive=False)
    
    # Set the action to send the message when the button is clicked
    send_button.click(fn=send_message, inputs=input_box, outputs=output_box)

# Launch the app
demo.launch()
