import torch

from scripts.compute_reward import compute_reward


def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """
    # bos = main_tokenizer.bos_token_id

    # inputs = torch.full((N,), fill_value=bos).to(device).unsqueeze(1)
    inputs = main_tokenizer(["It was"] * N, return_tensors='pt').to(device)
    candidates = main_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50, do_sample=True)
    samples = []
    for candidate in candidates:
        samples.append(main_tokenizer.decode(candidate.flatten().cpu().numpy().tolist()))
    rewards = compute_reward(reward_model, reward_tokenizer, samples)
    best = rewards.argmax()
    return samples[best]