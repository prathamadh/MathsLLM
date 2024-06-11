import gradio as gr
from huggingface_hub import InferenceClient
from model import *
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def evaluate_response(problem):
#     problem=b'what is angle x if angle y is 60 degree and angle z in 60 degree of a traingle'
    problem=problem.decode('utf-8')
    results, answers = [[],[]]
    messages = [{"role": "user", "content": problem  }] 
    query_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    raw_output = pipeline(query_prompt, max_new_tokens=2048, do_sample=True, temperature=0.9, return_full_text=False)

    raw_output = raw_output[0]['generated_text']
#     result_output, code_output = process_output(raw_output)
    return raw_output
    
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
# demo = gr.ChatInterface(
#     evaluate_response,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )

demo = gr.Interface(
        fn=evaluate_response,
        inputs=[gr.Textbox(label="Question")],
        outputs=gr.Textbox(label="Answer"),
        title="Question and Answer Interface",
        description="Enter a question."
    )

  
if __name__ == "__main__":
    demo.launch()