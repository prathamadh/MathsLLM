import re
import math
import random

from collections import defaultdict




def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)


import re
import sys
import subprocess

def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""

def process_code(code, return_shell_output=False):
    
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)

    if return_shell_output:
        code = code.replace('\n', '\n    ')
            # Add a try...except block
        code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
    
    if not return_shell_output:
        print(code)
    with open('code.py', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 7 ' + sys.executable + ' code.py'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        print(shell_output)
        if return_shell_output:
            if return_value=='FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value+='\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value))) % 1000
    except Exception as e:
        print(e,'shell_output')
        code_output = -1
    
    if return_shell_output:
        if code_output==-1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    
    
    return code_output


def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)

        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]

        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        
        else:
            result_output = round(float(eval(result_output))) % 1000
    
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output

from collections import defaultdict
from collections import Counter
def predict(problem):
    
    temperature = 0.9
    top_p = 3.0

    temperature_coding = 0.9
    top_p_coding = 3.0

   
    total_results = {}
    total_answers = {}
    best_stats = {}
    total_outputs = {}
    question_type_counts = {}
    starting_counts = (2,3)
    i = 0
    
    global n_repetitions,TOTAL_TOKENS,model,tokenizer,USE_PAST_KEY,NOTEBOOK_START_TIME,promplt_options,code,cot
    
    

    for jj in tqdm(range(n_repetitions)):
        best, best_count = best_stats.get(i,(-1,-1))
        if best_count>np.sqrt(jj):
            print("SKIPPING CAUSE ALREADY FOUND BEST")
            continue

        outputs = total_outputs.get(i,[])
        text_answers, code_answers = question_type_counts.get(i,starting_counts)
        results = total_results.get(i,[])
        answers = total_answers.get(i,[])  
        
        for _ in range(5):
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(0.2)
        
        try:
            ALREADY_GEN = 0
            code_error = None
            code_error_count = 0
            code_output = -1
            #initail_message = problem  + tool_instruction 
            counts = np.array([text_answers,code_answers])

            draw = choice(promplt_options, 1,
                          p=counts/counts.sum())

            initail_message = draw[0].format(problem,"{}")            
            prompt = f"User: {initail_message}"

            current_printed = len(prompt)
            print(f"{jj}_{prompt}\n")

            model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            input_len = len(model_inputs['input_ids'][0])

            generation_output = model.generate(**model_inputs, 
                                               max_new_tokens=TOTAL_TOKENS-ALREADY_GEN,
                                               return_dict_in_generate=USE_PAST_KEY,
                                               do_sample = True,
                                               temperature = temperature,
                                               top_p = top_p,
                                               num_return_sequences=1, stopping_criteria = stopping_criteria)

            if USE_PAST_KEY:
                output_ids = generation_output.sequences[0]
            else:
                output_ids = generation_output[0]
            decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"{decoded_output[current_printed:]}\n")
            current_printed += len(decoded_output[current_printed:])
            cummulative_code = ""
            
            stop_word_cond = False
            for stop_word in stop_words:
                stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)
                
                
            while (stop_word_cond) and (ALREADY_GEN<(TOTAL_TOKENS)):

                if (decoded_output[-len("```python"):]=="```python"):
                    temperature_inner=temperature_coding
                    top_p_inner = top_p_coding
                    prompt = decoded_output
                else:
                    temperature_inner=temperature
                    top_p_inner = top_p
                    try:
                        if (decoded_output[-len("``````output"):]=="``````output"):
                            code_text = decoded_output.split('```python')[-1].split("``````")[0]
                        else:
                            code_text = decoded_output.split('```python')[-1].split("```")[0]
                        

                        cummulative_code+=code_text
                        code_output, CODE_STATUS = process_code(cummulative_code, return_shell_output=True)
                        print('CODE RESULTS', code_output)

                        if code_error==code_output:
                            code_error_count+=1
                        else:
                            code_error=code_output
                            code_error_count = 0

                        if not CODE_STATUS:
                            cummulative_code = cummulative_code[:-len(code_text)]

                            if code_error_count>=1:
                                print("REPEATED ERRORS")
                                break

                    except Exception as e:
                        print(e)
                        print('ERROR PARSING CODE')
                        code_output = -1

                    if code_output!=-1:
                        if (decoded_output[-len(")\n```"):]==")\n```"):
                            prompt = decoded_output+'```output\n'+str(code_output)+'\n```\n'
                        else:
                            prompt = decoded_output+'\n'+str(code_output)+'\n```\n'
                    else:
                        prompt = decoded_output
                        cummulative_code=""
                model_inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                ALREADY_GEN =  len(model_inputs['input_ids'][0])-input_len

                if USE_PAST_KEY:
                    old_values = generation_output.past_key_values
                else:
                    old_values = None

                generation_output = model.generate(**model_inputs, 
                                                   max_new_tokens=TOTAL_TOKENS-ALREADY_GEN, 
                                                   return_dict_in_generate=USE_PAST_KEY,
                                                   past_key_values=old_values,
                                                   do_sample = True,
                                                   temperature = temperature_inner,
                                                   top_p = top_p_inner,
                                                   num_return_sequences=1, stopping_criteria = stopping_criteria)
                if USE_PAST_KEY:
                    output_ids = generation_output.sequences[0]
                else:
                    output_ids = generation_output[0]
                decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                print(f"\nINTERMEDIATE OUT :\n{decoded_output[current_printed:]}\n")
                current_printed+=len(decoded_output[current_printed:])
                
                stop_word_cond = False
                for stop_word in stop_words:
                    stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)
            if USE_PAST_KEY:
                output_ids = generation_output.sequences[0]
            else:
                output_ids = generation_output[0]

            raw_output = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
            #print(f"\n\nOutput :\n{raw_output}\n")                            
            result_output = process_text_output(raw_output)
            
            try:
                code_output = round(float(eval(code_output))) % 1000
            except Exception as e:
                print(e,'final_eval')
                code_output = -1
        except Exception as e:
            print(e,"5")
            result_output, code_output = -1, -1

        if code_output!=-1:
            outputs.append(code_output)
            code_answers+=1

        if result_output!=-1:
            outputs.append(result_output)
            text_answers+=1

        if len(outputs) > 0:
            occurances = Counter(outputs).most_common()
            print(occurances)
            if occurances[0][1] > best_count:
                print("GOOD ANSWER UPDATED!")
                best = occurances[0][0]
                best_count = occurances[0][1]
            if occurances[0][1] > 5:
                print("ANSWER FOUND!")
                break

        results.append(result_output)
        answers.append(code_output)
        
        best_stats[i] = (best, best_count) 
        question_type_counts[i] = (text_answers, code_answers)
        total_outputs[i] = outputs
        
        total_results[i] = results
        total_answers[i] = answers

        print("code_answers",code_answers-starting_counts[1],"text_answers",text_answers-starting_counts[0])
    return best_stats[0][0]
