import torch
from evaluate import load
from accelerate import Accelerator
from transformers import AutoTokenizer, GPT2LMHeadModel

from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

from argparse import ArgumentParser

from watermarking import detect_watermark, generate

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--n_sentences", type=int, default=128, help="Number of sentences to generate")
    parser.add_argument("--seq_len", type=int, default=200, help="Length of the generated sentences")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the generation")
    
    return vars(parser.parse_args())

@torch.no_grad()
def get_gpt2_perplexities(model, ids):
    """Returns the perplexity of the GPT2 model for the given tensor of indices.

    Args:
        model: The model to use for calculating perplexity.
        tensor: The tensor with the generated text indices.
    """
    perplexity = load("perplexity", module_type="metric")
    predictions = [model.tokenizer.decode(sentence)
                   for sentence in ids]
    return perplexity.compute(predictions=predictions, model_id='gpt2')["perplexities"]

class GPT2Wrapper(torch.nn.Module):
    """A wrapper around the GPT2 model to take ids as input and return logits as output."""

    def __init__(self):
        super(GPT2Wrapper, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs.logits


def main():
    # Device
    device = Accelerator().device
    
    # Plotting parameters
    args = parse_args()
    n_sentences = args["n_sentences"]
    seq_len = args["seq_len"]
    batch_size = args["batch_size"]
    
    # Model
    model = GPT2Wrapper().to(device)
    vocab_size = model.tokenizer.vocab_size
    
    # Prior text (BOS token)
    prior = (model.tokenizer.bos_token_id * torch.ones((n_sentences, 1))).long().to(device)
    
    # Collecting generations with and without watermark
    regular_ppls, regular_z_scores = [], []
    watermarked_ppls, watermarked_z_scores = [], []
    for i in tqdm(range(0, n_sentences, batch_size), desc="Generating sentences"):
        batch = prior[i:i+batch_size]
        
        # Regular sentences
        regular = generate(model, batch, max_length=seq_len, watermark=False)
        regular_ppls.extend(get_gpt2_perplexities(model, regular))
        regular_z_scores.extend(detect_watermark(regular, vocab_size).tolist())
        
        # Watermarked sentences
        watermarked = generate(model, batch, max_length=seq_len, watermark=True, gamma=0.1, delta=5)
        watermarked_ppls.extend(get_gpt2_perplexities(model, watermarked))
        watermarked_z_scores.extend(detect_watermark(watermarked, vocab_size).tolist())
    
    # Scatter plot of perplexity vs z-score
    plt.scatter(regular_ppls, regular_z_scores, label="Regular")
    plt.scatter(watermarked_ppls, watermarked_z_scores, label="Watermarked")
    plt.legend()
    plt.title("Perplexity vs Z-score")
    plt.xlabel("Perplexity")
    plt.ylabel("Z-score")
    plt.savefig("perplexity_vs_zscore.png")
    plt.show()
    print("Program completed successfully!")


if __name__ == "__main__":
    main()
