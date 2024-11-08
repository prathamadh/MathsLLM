import gradio as gr
import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from codeexecutor import get_majority_vote,type_check,postprocess_completion,draw_polynomial_plot
import base64
from io import BytesIO

import re
import os
# Define the model and tokenizer loading
model_prompt = "Explain and solve the following mathematical problem step by step, showing all work: "
tokenizer = AutoTokenizer.from_pretrained("AI-MO/NuminaMath-7B-TIR")
model_path = snapshot_download(repo_id="Makima57/deepseek-math-Numina")
generator = ctranslate2.Generator(model_path, device="cpu", compute_type="int8")
iterations = 4
test=False

# Function to generate predictions using the model
def get_prediction(question):
    if test==True:
        text="Solve the following mathematical problem: what is  sum of polynomial 2x+3 and 3x?\n### Solution: To solve the problem of summing the polynomials \\(2x + 3\\) and \\(3x\\), we can follow these steps:\n\n1. Define the polynomials.\n2. Sum the polynomials.\n3. Simplify the resulting polynomial expression.\n\nLet's implement this in Python using the sympy library.\n\n```python\nimport sympy as sp\n\n# Define the variable\nx = sp.symbols('x')\n\n# Define the polynomials\npoly1 = 2*x + 3\npoly2 = 3*x\n\n# Sum the polynomials\nsum_poly = poly1 + poly2\n\n# Simplify the resulting polynomial\nsimplified_sum_poly = sp.simplify(sum_poly)\n\n# Print the simplified polynomial\nprint(simplified_sum_poly)\n```\n```output\n5*x + 3\n```\nThe sum of the polynomials \\(2x + 3\\) and \\(3x\\) is \\(\\boxed{5x + 3}\\).\n"  

        return text
    input_text = model_prompt + question
    input_tokens = tokenizer.tokenize(input_text)
    results = generator.generate_batch(
        [input_tokens],
        max_length=512,
        sampling_temperature=0.7,
        sampling_topk=40,
    )
    output_tokens = results[0].sequences[0]
    predicted_answer = tokenizer.convert_tokens_to_string(output_tokens)
    return predicted_answer

# Function to parse the prediction to extract the answer and steps
def parse_prediction(prediction):
    lines = prediction.strip().split('\n')
    answer = None
    steps = []
    # for line in lines:
    #     # Check for "Answer:" or "answer:"
    #     match = re.match(r'^\s*(?:Answer|answer)\s*[:=]\s*(.*)', line)
    #     if match:
    #         answer = match.group(1).strip()
    #     else:
    #         answer=lines[-1].strip()
    # if answer is None:
    #     # If no "Answer:" found, assume last line is the answer
    answer = lines[-1].strip()
    steps = lines
    steps_text = '\n'.join(steps).strip()
    return answer, steps_text

# Function to perform majority voting and get steps
def majority_vote_with_steps(question, num_iterations=4):
    all_predictions = []
    all_answers = []
    steps_list = []
    plot_file=None

    for _ in range(num_iterations):
        prediction = get_prediction(question)
        answer, success = postprocess_completion(prediction, return_status=True, last_code_block=True)
        print(answer,success)

        if success:
            all_predictions.append(prediction)
            all_answers.append(answer)
            steps_list.append(prediction)
        else:
            answer, steps = parse_prediction(prediction)
            all_predictions.append(prediction)
            all_answers.append(answer)
            steps_list.append(steps)
    majority_voted_ans = get_majority_vote(all_answers)
    if success:
        
        expression = majority_voted_ans
        if type_check(expression) == "Polynomial":
            plot_file = draw_polynomial_plot(expression)
    else:
        plot_file = "polynomial_plot.png"

    # Find the steps corresponding to the majority voted answer
    for i, ans in enumerate(all_answers):
        if ans == majority_voted_ans:
            steps_solution = steps_list[i]
            answer = parse_prediction(steps_solution)
            break
    else:
        answer = majority_voted_ans
        steps_solution = "No steps found"

    return answer, steps_solution, plot_file

# Function to handle chat-like interaction and merge plot into chat history
def chat_interface(history, question):
    final_answer, steps_solution, plotfile = majority_vote_with_steps(question, iterations)
    
    # Convert the plot image to base64 for embedding in chat (if plot exists)
    if plotfile:
        history.append((question, f"Answer: \n{steps_solution}"))
        
        with open(plotfile, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            image_data = f'<img src="data:image/png;base64,{base64_image}" width="300"/>'
            history.append(("", image_data)) 
    else:
        history.append(("MathBot", f"Answer: \n{steps_solution}"))
    
    return history 

custom_css = """
#math_question label {
    font-size: 20px;  /* Increase label font size */
    font-weight: bold; /* Optional: make the label bold */
}

#math_question textarea {
    font-size: 20px;  /* Increase font size */
}
"""
# Gradio app setup using Blocks
with gr.Blocks(css=custom_css) as interface:
    chatbot = gr.Chatbot(label="Chat with MathBot", elem_id="chat_history",height="70vh")
    math_question = gr.Textbox(label="Your Question", placeholder="Ask a math question...", elem_id="math_question")
    
    math_question.submit(chat_interface, inputs=[chatbot, math_question], outputs=[chatbot])

interface.launch()