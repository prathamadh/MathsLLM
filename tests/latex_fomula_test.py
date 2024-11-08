import gradio as gr

# Define a function that will be called when the send button is clicked or the textbox is submitted
def chat_interface(chatbot_history, question):
    chatbot_history.append(("You", question))  # Add the user's question to the chatbot history
    
    # Generate a response with LaTeX (e.g., a math formula)
    answer = r"Here is a math formula: $$E = mc^2$$"+question  # LaTeX syntax within the string
    chatbot_history.append(("MathBot", answer))  # Add the bot's response with LaTeX
    
    return chatbot_history

# Custom CSS to style the textarea and make the send button small with an icon
custom_css = """
#math_question label {
    font-size: 20px;  /* Increase label font size */
    font-weight: bold; /* Optional: make the label bold */
}

#math_question textarea {
    font-size: 20px;  /* Increase font size */
}

#send_button {
    font-size: 12px;  /* Smaller font size for the button */
    padding: 4px 8px;  /* Smaller padding for a compact button */
    width: 40px;  /* Set smaller width of the button */
    height: 40px;  /* Set smaller height of the button */
    margin-top: 10px;  /* Add some space above the button */
    background-color: #4CAF50; /* Optional: background color */
    color: white; /* Optional: white color for the icon */
    border-radius: 50%; /* Make the button circular */
}
"""

# Gradio app setup using Blocks
with gr.Blocks(css=custom_css) as interface:
    chatbot = gr.Chatbot(label="Chat with MathBot", elem_id="chat_history", height="70vh")
    with gr.Row():
        math_question = gr.Textbox(label="Your Question", placeholder="Ask a math question...", elem_id="math_question")
        send_button = gr.Button("🔽", elem_id="send_button")  # Create a small button with an arrow icon

    # Set the actions for submitting through the textbox or clicking the send button
    math_question.submit(chat_interface, inputs=[chatbot, math_question], outputs=[chatbot])
    send_button.click(chat_interface, inputs=[chatbot, math_question], outputs=[chatbot])

# Launch the interface
interface.launch(share=False, debug=True)
