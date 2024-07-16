---
layout: default
title:
permalink: /blogs/ppl/
---



This mini blog answers:
1. What is the famous  the famous perplexity score?
2. How is it implemented, especially in Hugging Face?

## Small wiki/math:

Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. If we have a tokenized sequence $$X = (x_0, x_1, \ldots, x_t)$$, then the perplexity of $$X$$ is,

$$
\text{ppl}(X) = \exp 
\left\{ 
-\frac{1}{t} \sum_{i=1}^{t} \log p_{\theta}(x_i \mid x_{< i}) 
\right\}
$$

## Code:

The following code snippet computes the perplexity score of GPT2-small on Lambada:

<details>
	<summary> Click show pre-processing (loading the model, dataset, and configs)</summary>
<div markdown="1">
```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

import torch
from tqdm import tqdm
import numpy as np  

device = "cuda"

model  = GPT2LMHeadModel.from_pretrained(f"./hf/73150").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

test = load_dataset("lambada", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
```
</div>

</details>

The main loop:

```python
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  
	# may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
```

Let's dissect the main loop. Matching the equation above and `ppl = torch.exp(torch.stack(nlls).mean())`, `nlls[i]` must represent the $$\log p_{\theta}(x_i \mid x_{< i})$$.


Question is, what is the `outputs = model(input_ids, labels=target_ids)`. 


The variable `outputs` is of type `transformers.modeling_outputs.CausalLMOutputWithCrossAttentions`, and has three keys:

1. `outputs.loss`: a single scaler, it represents exactly the quantity $$\log p_{\theta}(x_i \mid x_{< i})$$.
2. `outputs.logits`: this is the actual output matrix of the LM, is has a shape of `[1, seq_len, vocab_len]`.
3. `past_key_values` will ignore for now.


How to compute the `loss` from the `logits`:

We have:

```python
outputs = model(input_ids, labels=target_ids)
# input_ids >> tensor([[  257,  1598,  7815,  ...,  1175, 32002,   379]], device='cuda:0')
# target_ids >> tensor([[-100, -100, -100,  ..., -100, -100,  379]], device='cuda:0')
```

Hence, for each run:

* We are only interested in the model prediction for the **BEFORE** last token, which is **32002** in this example.
* We need to look at `outputs[0,-2]` and not `outputs[0,-1]`.
* `outputs[0, -2]` has a `[1, vocab_size]` shape. And, `outputs[0, -2][379]` would be represent exactly how much weight does the model think that the next token after 32002 would be **379**.
* `outputs[0, -2]` is not normalized. Hence, it should be softmaxe'd first. 

## References:

* [Hugging Face blog](https://huggingface.co/docs/transformers/en/perplexity)
* [The gradient blog](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)



