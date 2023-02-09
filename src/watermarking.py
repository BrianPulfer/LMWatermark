"""Implementation of watermarking for language models as proposed in the paper

            "A Watermark for Large Language Models"
by Kirchenbauer, Geiping et. al. (https://arxiv.org/abs/2301.10226v2).
"""

import torch
import numpy as np


def default_hash_fn(tensor):
    """Returns the hash of the given tensor by converting it to a string first.

    Args:
        tensor: The tensor to hash.

    Returns:
        The hash of the tensor.
    """
    return hash(str(tensor))


@torch.no_grad()
def generate(model, prior_tokens, max_length=200, watermark=True, gamma=0.5, delta=2, hash_function=default_hash_fn):
    """Generates text with the given model. Optionally, the text can be (soft) watermarked.

    Args:
        model: The model which outputs logits for the next token.
        prior_tokens: The input tensor from which the model starts generating of shape (B, T).
        max_length: The maximum length of the generated text. Default is 100.
        gamma: The proportion of the green list. Default is 0.5.
        delta: The hardness parameter. Default is 20.
        hash_function: The function to use for hashing. Default is default_hash_fn.
        
    Returns:
        The generated text of shape (B, T).
    """
    B = prior_tokens.shape[0]

    generated_tokens = prior_tokens
    for _ in range(max_length):
        # Getting logits
        l_t = model(generated_tokens)[:, -1, :]

        if watermark:
            # Seeding generators based on previous token
            seeds = [hash_function(generated_tokens[i, -1]) for i in range(B)]
            generators = [torch.Generator().manual_seed(seed)
                          for seed in seeds]

            # Increasing probability of green list indices
            vs = l_t.shape[-1]  # Vocabulary size
            gls = int(gamma * vs)  # Green list size
            gli = torch.stack([torch.randperm(vs, generator=generators[i])[:gls]
                               for i in range(B)])  # Green list indices

            for i in range(B):
                l_t[i, gli[i]] = l_t[i, gli[i]] + delta

        # Sampling from the distribution
        l_t = torch.softmax(l_t, dim=-1)
        next_tokens = torch.multinomial(l_t, 1)
        generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

    return generated_tokens


def detect_watermark(ids, vocab_size, gamma=0.5, hash_function=default_hash_fn):
    """Returns the probability that a text was created by a Language Model with watermarking.

    Args:
        ids: The tensor with the generated text indices of shape (B, T).
        gamma: The proportion of the green list. Default is 0.5.
        delta: The hardness parameter. Default is 20.
        hash_function: The function used for watermarking. Default is default_hash_fn.
        
    Returns:
        The z-statistic of the watermarking probability.
    """
    B, T = ids.shape
    ids = ids.cpu()
    gls = int(gamma * vocab_size)  # Green list size
    in_green_list = torch.zeros(B, dtype=torch.float32) # Number of tokens in the green list

    for i in range(T-1):
        # Seeding generators based on previous token
        seeds = [hash_function(ids[j, i]) for j in range(B)]
        generators = [torch.Generator().manual_seed(seed) for seed in seeds]

        # Increasing probability of green list indices
        gli = torch.stack([torch.randperm(vocab_size, generator=generators[i])[:gls]
                           for i in range(B)])  # Green list indices

        # Counting tokens that are in the green list and adding to the total
        in_green_list += torch.tensor([ids[sequence, i+1] in gli[sequence]
                                      for sequence in range(B)]).int()
        
    z = (in_green_list - gamma * T) / np.sqrt(T*gamma*(1-gamma))
    return z
