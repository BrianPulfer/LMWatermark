# Watermarking for language models
## Description
Implementation of the watermarking technique proposed in [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226v2)
by **Kirchenbauer** & **Geiping** et. al.

## Usage
Generating a (soft) watermarked text with your language model is as easy as:

```python
from watermark import generate

# Loading the model
model = load_my_model().eval().to(device)

# Creating prior text
prior = torch.randint(0, vocab_size, (batch_size, 1)).to(device)

# Generating the watermarked text
watermarked = generate(model, prior, max_length=200, watermarked=True, gamma=0.5, delta=2)
```

Verfiying if a text was watermarked can be done as follows:

```python
from watermarking import detect_watermark

# Text is a (B, T) tensor of idxs
z_score = detect_watermark(text, vocabulary_size, gamma=0.5)

if (z_score >= threshold):
    print("Text has been AI-generated.")
```


For more information, refer to [this example](./../src/main.py).


## License
The code is released with the [MIT license](./../LICENSE).