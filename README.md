# IntelliMath Solver

## Introduction
IntelliMath Solver is an advanced chatbot designed for solving mathematical problems using state-of-the-art language models. This project integrates Large Language Models (LLMs) with tools like Gradio to provide users with a responsive interface where they can query math-related questions and receive detailed solutions. The project aims to enhance the capabilities of AI in accurately solving complex math questions, particularly in the field of algebra, calculus, and more.
Large language models (LLMs) are a rapidly developing field in artificial intelligence that has achieved notable progress in a number of areas, including general mathematics. Nonetheless, a significant obstacle still exists: these models frequently falter when faced with complex mathematical problems requiring specialized knowledge and cognitive skills. The availability of large training datasets makes LLMs more accurate in broad mathematical queries, but their performance declines in issues requiring domain-specific knowledge.
This research develops an advanced reasoning-oriented mathematical LLM with the goal of addressing this constraint. Our aim is to improve LLMs' mathematical problem-solving abilities by utilizing insights from neuroscience and cutting-edge technologies.
To achieve this, we will investigate three core sub-problems, to assess whether current LLMs can effectively tackle and solve a variety of mathematical problems with high accuracy, to determine if LLMs can produce precise Sympy code for solving mathematical problems and evaluate the efficiency of this process in terms of time and accuracy. And explore the potential for developing an LLM that emulates human-like mathematical reasoning. This involves integrating advanced concepts such as world models, neurosymbolic approaches, and innovative architectures to enhance the model’s ability to generalize and reason with numbers and formulas.
By solving these sub-problems, we hope to develop a model that can use chain-of-thought techniques and specialized treatment of values within formulas to perform more rigorous mathematical reasoning. In order to advance the state of the art in AI-driven reasoning, this research aims to push the limits of LLM capabilities in the field of mathematical problem-solving.

## Goals
Create an AI-powered chatbot capable of solving a wide variety of mathematical questions.
Provide a user-friendly interface for querying and receiving solutions.
Continuously improve model accuracy and efficiency for real-world applications.
Facilitate easy interaction with mathematical models through Gradio interfaces.
## Contributors
Pratham Adhikari
Shashwot Pradhan
Prashant Subedi
## Project Architecture

In our quest to enhance the mathematical reasoning capabilities of large language models (LLMs), we are inspired by the way the human brain processes and reasons through thoughts. Specifically, we aim to implement an architecture informed by the principles outlined in the paper "Dual-process theories of thought as potential architectures for developing neuro-symbolic AI models." This approach involves a three-stage process designed to emulate the brain's reasoning pathways.

Stage 1: Initial Solution Generation 
The first stage involves using the LLM to generate a potential solution to the given mathematical problem. This stage leverages the model's ability to process and produce answers based on its extensive training data. The goal is to produce an initial solution that can be further analyzed and refined in subsequent stages. We will train a Deepseek math model or Gemma model.

Stage 2: Statistical Verification and Execution In the second stage
 the proposed solution undergoes a statistical verification process. This involves checking the accuracy of the answer and executing any associated code, such as Sympy scripts, to ensure the solution is correct. If the answer meets the required criteria, it is accepted; otherwise, it is flagged for further scrutiny. we can use Gaussian distribution to check for consistency.

Stage 3: Holistic, Abstract Problem Analysis and Deep analyzing
The third stage is where the model takes a holistic or abstract view of the problem. This involves a deeper analysis to identify any conceptual inconsistencies and to perform a thorough examination of the problem. By approaching the problem from a higher-level perspective, the model can detect and correct errors that were not apparent in the initial stages. This stage is crucial for developing a robust understanding and reasoning capability akin to human thought processes.

Our implementation will focus on fully developing Stage 1 and Stage 2, and establishing the foundation for Stage 3. Stage 3, in particular, is a highly ambitious area of research with the potential to bring us closer to achieving strong AI or Artificial General Intelligence (AGI). By exploring various novel architectures, knowledge representation techniques, and innovative ideas, we aim to significantly improve the model's accuracy and reasoning capabilities.

In conclusion, this project aims to push the boundaries of what LLMs can achieve in mathematical reasoning by mimicking the dual-process theories of human thought. Through the careful implementation of these three stages, we hope to create a model that not only solves mathematical problems but does so with a depth of understanding and accuracy that brings us closer to the vision of AGI.
Model Backend: Utilizes models like 'AI-MO/NuminaMath-7B-TIR' and 'Makima57/deepseek-math-Numina' for answering math queries.
Gradio Interface: Offers an intuitive UI for users to input questions and receive answers.
Evaluation Module: Compares the predicted results with correct answers and displays them in a user-friendly format.
Environment Setup: Virtual environment management via pip-tools and code formatting/linting using pre-commit.

![LLM Architecture](https://github.com/fuseai-fellowship/IntelliMath-Solver/blob/shashwot/image/llm.png) 
# Status
The project is currently completed with the core functionalities in place, including model integration and basic Gradio interface.
## Known Issue
There might be latency in providing responses for complex questions due to model size and computation requirements.
## High Level Next Steps
Optimize the models to reduce latency and improve accuracy.
Enhance UI/UX for a better user experience.
Integrate additional models or features to support a broader range of mathematical problems.

# Usage
You can interact with the IntelliMath Solver via the Gradio interface that has been published on Hugging Face space by simply entering a math query. The chatbot will process the input and display both the generated and correct answers for evaluation.
## Installation
To begin this project, use the included `Makefile`

#### Creating Virtual Environment

This package is built using `python-3.8`. 
We recommend creating a virtual environment and using a matching version to ensure compatibility.

#### pre-commit

`pre-commit` will automatically format and lint your code. You can install using this by using
`make use-pre-commit`. It will take effect on your next `git commit`

#### pip-tools

The method of managing dependencies in this package is using `pip-tools`. To begin, run `make use-pip-tools` to install. 

Then when adding a new package requirement, update the `requirements.in` file with 
the package name. You can include a specific version if desired but it is not necessary. 

To install and use the new dependency you can run `make deps-install` or equivalently `make`

If you have other packages installed in the environment that are no longer needed, you can you `make deps-sync` to ensure that your current development environment matches the `requirements` files. 

## Usage Instructions


# Data Source
The IntelliMath Solver uses data from several well-known mathematical problem-solving sources, which have been merged to create a larger and more comprehensive dataset. The key sources are:

AMIO Parsed "Art of Problem Solving" Website:

The Art of Problem Solving (AoPS) website is a popular platform for learning advanced mathematics, particularly for students preparing for math competitions such as the Mathematical Olympiads.
Problems and solutions from AoPS were parsed using the AMIO (Artificial Mathematical Intelligence Organizer) system. These parsed problems, along with their detailed solutions, provide high-quality, real-world mathematical challenges suitable for model training and testing.
Mathematical Olympiads Problems with Solutions:

This dataset includes problems from various Mathematical Olympiads (such as the International Mathematical Olympiad) along with their solutions. These Olympiad problems are known for their difficulty and require advanced mathematical reasoning and creative problem-solving skills.
By incorporating these problems, the dataset ensures that the model is capable of solving both elementary and advanced level mathematical queries.
lighteval / MATH Dataset:

The lighteval / MATH dataset, also known as Hendrycks’ MATH Dataset, is a curated set of math problems aimed at evaluating machine learning models' mathematical reasoning abilities.
This dataset covers a wide range of mathematical topics, including algebra, geometry, calculus, and probability. It was developed to evaluate models' abilities to solve mathematics problems of varying difficulty.
Merged Datasets:

The above-mentioned datasets were merged to form a larger, unified dataset. By combining data from different sources, we increase the variety and complexity of math problems available for model training and evaluation.
The merged dataset includes a diverse range of problem types, from simple arithmetic to challenging Olympiad-level problems. This ensures that the IntelliMath Solver can handle a broad spectrum of mathematical queries, improving the model's robustness and generalization.
In summary, by leveraging these high-quality data sources, the IntelliMath Solver is trained on a diverse set of math problems, ranging from elementary to advanced levels, ensuring that it can effectively solve a wide range of mathematical tasks.
## Code Structure
## Artifacts Location
All our prototypes and model are located in Hugging face hub 
some of them are as follows : 
Bart Model : https://huggingface.co/Pra-tham/results 

LLM : https://huggingface.co/Pra-tham/quant_deepseekmath 

our generated dataset in `data` folder
# Results
The chatbot will provide solutions for math problems in real-time using the integrated models. All generated answers and correct ones are displayed for easy evaluation.
## Metrics Used
In the IntelliMath Solver project, multiple evaluation metrics are used to assess the quality of generated answers from the models. Each of these metrics offers a different perspective on the accuracy and relevance of the predicted results.

Cosine Similarity: Cosine similarity is a measure that calculates the cosine of the angle between two vectors in a multi-dimensional space. In this context, the vectors represent the embeddings (numerical representations) of the predicted and true answers.
 Cosine similarity is particularly useful for comparing text-based answers, as it captures the semantic similarity between them even when the exact wording differs. In mathematical problems, two answers could have different formats but still be equivalent (e.g., 1/2 and 0.5), and cosine similarity helps capture this nuance.

Accuracy: Accuracy measures the percentage of correct answers generated by the model relative to the total number of questions asked. It is the simplest and most direct metric. Measures how many correct answers were generated compared to the total questions.
While accuracy is a basic metric, it provides a straightforward measure of the model’s performance. However, it may not always reflect the nuanced performance of models in more complex scenarios where partial correctness or alternative correct answers are possible.

Token matching: Token matching is an evaluation technique that breaks down both the predicted and correct answers into individual tokens (words, numbers, or symbols) and compares them. The predicted and correct answers are tokenized into components such as numbers, variables, and operators (in mathematical contexts).This approach focuses on matching key components of the answer rather than requiring an exact string match.
Token matching allows for a more flexible evaluation, especially in mathematics, where answers can be expressed in different but correct forms (e.g., x = 3 and 3 = x). By comparing individual tokens, this metric ensures that the essential parts of the answer are correct, even if the format differs.

Majority voting: Majority voting is an ensemble method where multiple models or versions of the same model generate answers, and the final answer is chosen based on the majority. Multiple models generate potential answers to a given math problem. The answer that appears the most frequently among the generated results is selected as the final prediction.
Majority voting helps mitigate errors from individual models and increases overall reliability. By aggregating predictions from multiple models, the final result is more likely to be accurate, particularly when the models have complementary strengths.
By using a combination of these metrics, IntelliMath Solver can provide a comprehensive evaluation of its performance, considering exact correctness (accuracy), component-level correctness (token matching), semantic similarity (cosine similarity), and robustness through ensemble methods (majority voting).
## Evaluation Results
In this section, we discuss the performance of the IntelliMath Solver based on various models and evaluation methods. The results highlight the superiority of certain approaches over others for solving mathematical problems, as well as the development of a new validation technique for evaluating model output.

1. DeepSeek Math is better than Gemma for solving math problems
The DeepSeek Math model has demonstrated superior performance over the Gemma model in solving mathematical problems. This is based on several factors:

Mathematical Focus: DeepSeek Math has been specifically trained on large datasets of mathematical questions and solutions, giving it a better understanding of both the structure and reasoning needed to solve complex math problems. Gemma, while capable in general NLP tasks, lacks the fine-tuning needed for advanced mathematical reasoning.

Handling Complex Queries: DeepSeek Math excels in handling complex multi-step mathematical problems, such as calculus, algebra, and geometry. It has been optimized for reasoning through steps in a solution, whereas Gemma tends to struggle with such tasks, often returning incomplete or incorrect answers for more challenging queries.

Higher Accuracy: The overall accuracy of DeepSeek Math, as measured through token matching and correctness, consistently outperforms Gemma. For example, DeepSeek Math has been observed to achieve an accuracy of over 85% on benchmark math datasets, while Gemma's performance lags behind, particularly in advanced problem sets.

In conclusion, for the task of solving mathematical problems, DeepSeek Math provides better model outputs and is better aligned with the objectives of IntelliMath Solver than Gemma.

2. SymPy Parsing is better than Cosine Similarity for mathematical evaluation
SymPy parsing is a more robust evaluation method than cosine similarity for mathematical problems because it focuses on the mathematical correctness of the answer, rather than just comparing text-based representations of the solutions.

How SymPy Parsing Works: SymPy, a Python library for symbolic mathematics, parses both the predicted and correct answers as mathematical expressions. It then checks for their equivalence. This is important because mathematically equivalent expressions can have different text representations (e.g., x^2 - 4 and (x - 2)(x + 2) are mathematically the same but appear different as strings).

Advantages over Cosine Similarity:

Mathematical Equivalence: Cosine similarity focuses on comparing vector representations of answers, which works well for general text-based problems but falls short when dealing with mathematically equivalent expressions that are formatted differently. For example, 1/2 and 0.5 would not be recognized as equivalent by cosine similarity unless their representations are identical.
Higher Precision: SymPy parsing ensures that even if the format or syntax differs between the predicted and correct answers, they are still evaluated as correct if they are mathematically the same. This leads to higher precision and more meaningful validation for math problems.
In short, SymPy parsing is more suitable for evaluating mathematical answers, as it ensures correctness based on mathematical equivalence rather than relying on similarity measures that can overlook critical differences.

This comprehensive approach to evaluation ensures that the IntelliMath Solver is not only accurate in solving math problems but also rigorously validated, leveraging advanced parsing techniques and hybrid metrics for better accuracy and reliability.
