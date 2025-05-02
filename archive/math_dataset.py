from datasets import load_dataset, DatasetDict, Dataset
dd: DatasetDict = load_dataset("sl-alex/openai-prm800k-solutions-only")
dd: DatasetDict = load_dataset("sl-alex/openai-prm800k-stepwise-best")
dd: DatasetDict = load_dataset("sl-alex/openai-prm800k-stepwise-critic")

d: Dataset = dd['train']

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request. Show your reasoning step-by-step, and indicate your final answer under a heading titled Answer.

### Instruction:
{instruction}

### Response:
{response_history}"""

answer_template = """{response}

# Answer:
{answer}"""

for record in zip(
    d['instruction'],
    d['responses'],
    d['next_response'],
    d['answer'],
    d['is_human_response'],
    d['is_solution'],
    d['is_preferred_response'],
    d['rating']
):
    instruction, responses, next_response, answer, is_human_response, is_solution, is_preferred_response, rating = record
    prompt = prompt_template.format(
    instruction=instruction,
    response_history=''.join((f'{response}\n' for response in responses)),
    )
    completion=next_response if answer is None else answer_template.format(response=next_response, answer=answer)
    print(f'Prompt:\n<{prompt}>')
    print(f'Completion:\n<{completion}>')