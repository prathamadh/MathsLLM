import os
import re
import subprocess
import tempfile
import multiprocessing
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


class PythonREPL:
    def __init__(self, timeout=15):
        self.timeout = timeout

    @staticmethod
    def _run_code(temp_file_path):
        result = subprocess.run(
            ["python3", temp_file_path],
            capture_output=True,
            check=False,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            error_msg = result.stderr.strip()
            msgs = error_msg.split("\n")
            new_msgs = []
            want_next = False
            for m in msgs:
                if "Traceback" in m:
                    new_msgs.append(m)
                elif m == msgs[-1]:
                    new_msgs.append(m)
                elif temp_file_path in m:
                    st = m.index('"/') + 1 if '"/' in m else 0
                    ed = m.index(temp_file_path) + 1 if temp_file_path in m else None
                    clr = m[st:ed] if not ed else m[st:]
                    m = m.replace(clr, "")
                    new_msgs.append(m)
                    want_next = True
                elif want_next:
                    new_msgs.append(m)
                    want_next = False
            return False, "\n".join(new_msgs).strip()

    def __call__(self, query):
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            if "#" in query[-1]:
                query[-1] = query[-1].split("#")[0]
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)

            with multiprocessing.Pool(1) as pool:
                result = pool.apply_async(self._run_code, (temp_file_path,))
                try:
                    success, output = result.get(self.timeout)
                except multiprocessing.TimeoutError:
                    pool.terminate()
                    return False, f"Timed out after {self.timeout} seconds."
        return success, output


def execute_completion(executor, completion, return_status, last_code_block):
    executions = re.findall(r"```python(.*?)```", completion, re.DOTALL)
    if len(executions) == 0:
        return completion, False if return_status else completion
    if last_code_block:
        executions = [executions[-1]]
    outputs = []
    successes = []
    for code in executions:
        success = False
        for lib in ("subprocess", "venv"):
            if lib in code:
                output = f"{lib} is not allowed"
                outputs.append(output)
                successes.append(success)
                continue
        try:
            success, output = executor(code)
        except TimeoutError as e:
            print("Code timed out")
            output = e
        if not success and not return_status:
            output = ""
        outputs.append(output)
        successes.append(success)
    output = str(outputs[-1]).strip()
    success = successes[-1]
    if return_status:
        return output, success
    return output ,False


def postprocess_completion(text, return_status, last_code_block):
    executor = PythonREPL()
    result = execute_completion(executor, text, return_status=return_status, last_code_block=last_code_block)
    del executor
    return result


def get_majority_vote(answers):
    if not len(answers):
        return 0
    c = Counter(answers)
    value, _ = c.most_common()[0]
    return value


def type_check(expr_str):
       
       
        expr = sp.sympify(expr_str)
        
        # Check if the expression is a real number
        if expr.is_real:
            return "Real"

        # Check if the expression is a complex number
        if expr.is_complex:
            return "Complex"

        # Check if the expression is a polynomial
        if expr.is_polynomial():
            return "Polynomial"

        # Otherwise, classify as other
        return "Other"


def draw_polynomial_plot(expression):
    try:
        x = sp.symbols('x')
        poly_expr = sp.sympify(expression)  # Convert input to sympy expression
        poly_lambda = sp.lambdify(x, poly_expr, 'numpy')

        # Create the plot
        x_vals = np.linspace(-10, 10, 400)
        y_vals = poly_lambda(x_vals)

        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.title('Polynomial Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # Save the plot to a file
        plot_filename = "polynomial_plot.png"
        plt.savefig(plot_filename)
        plt.close()

        return plot_filename 
    except Exception as e:
        print(f"Error in draw_polynomial_plot: {e}")
        return None


    