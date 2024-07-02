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


# import gradio as gr
# # from huggingface_hub import InferenceClient

# """
# For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# """
# # client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
# # from accelerate import infer_auto_device_map as iadm

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# from transformers import BitsAndBytesConfig
# from tqdm import tqdm
# import os 


# USE_PAST_KEY = True
# import gc
# torch.backends.cuda.enable_mem_efficient_sdp(False)

# from transformers import (
#     AutoModelForCausalLM, 
#     AutoTokenizer, 
#     AutoConfig,
#     StoppingCriteria,
#     set_seed
# )

# n_repetitions = 1 
# TOTAL_TOKENS = 2048 

# MODEL_PATH = "Pra-tham/quant_deepseekmath"
#     #"/kaggle/input/gemma/transformers/7b-it/1"
    
# # DEEP = True
# import torch

# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# import transformers



# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# model = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             device_map="cpu",
#             torch_dtype="auto",
#             trust_remote_code=True, 

#         )
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype='auto',
#     device_map='cpu',
# )
# from transformers import StoppingCriteriaList

# class StoppingCriteriaSub(StoppingCriteria):
#         def __init__(self, stops = [], encounters=1):
#             super().__init__()
#             # self.stops = [stop.to("cuda") for stop in stops]
#             self.stops = stops

#         def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#             for stop in self.stops:
#                 last_token = input_ids[0][-len(stop):]
#                 if torch.all(torch.eq(stop,last_token)):
#                     return True
#             return False


# stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output"] #,  
# stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# code = """Below is a math problem you are to solve (positive numerical answer):
# \"{}\"
# To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be positive integer, not an algebraic expression!
# Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.

# Approach:"""


# cot = """Below is a math problem you are to solve (positive numerical answer!):
# \"{}\"
# Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""

# promplt_options = [code,cot]

# import re
# from collections import defaultdict
# from collections import Counter

# from numpy.random import choice
# import numpy as np

# tool_instruction = '\n\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numercal answer would always be an integer.'


# #tool_instruction = " The answer should be given as a non-negative modulo 1000."
# #tool_instruction += '\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.'

# demo = gr.Interface(
#         fn=predict,
#         inputs=[gr.Textbox(label="Question")],
#         outputs=gr.Textbox(label="Answer"),
#         title="Question and Answer Interface",
#         description="Enter a question."
#     )

# import subprocess

# def reboot_system():
#     try:
#         # Execute the reboot command
#         subprocess.run(['sudo', 'reboot'], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error occurred while trying to reboot the system: {e}")
  
# if __name__ == "__main__":
#     if os.path.exists("temp.txt"):
#         os.remove("temp.txt")
#         reboot_system()
#     demo.launch()