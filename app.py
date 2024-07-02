import gradio as gr
# from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
# from accelerate import infer_auto_device_map as iadm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id




def evaluate_response(problem):
#     problem=b'what is angle x if angle y is 60 degree and angle z in 60 degree of a traingle'
    problem=problem+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
    messages = [
        {"role": "user", "content": problem}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
#     result_output, code_output = process_output(raw_output)
    return result
    
# def respond(
#     evaluate_response,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response

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