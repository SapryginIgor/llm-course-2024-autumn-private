def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    # <ВАШ КОД ЗДЕСЬ>
    subject = sample['subject']
    a,b,c,d = sample['choices']
    question = sample['question']
    prompt = f"""The following are multiple choice questions (with answers) about {subject}.
{question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""
    return prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    # <ВАШ КОД ЗДЕСЬ>

    # return <ВАШ КОД ЗДЕСЬ>
    prompt = ""
    for example in examples:
        prompt += create_prompt(example)
        letter = chr(ord('A') + example['answer'])
        prompt += f" {letter}"
        if add_full_example:
            prompt += f". {example['choices'][example['answer']]}"
        prompt += '\n\n'
    prompt += create_prompt(sample)
    return prompt

