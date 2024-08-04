---
layout: default
title:
permalink: /blogs/ppl/
---



This mini blog answers:

1. What is the perplexity score.
2. How is it implemented, e.g., with Hugging Face models.
3. Can we do better than perplexity?

## Quick wiki:

Given a  tokenized sequence $$X = (x_0, x_1, \ldots, x_t)$$ and an autoregressive model $$p_{\theta}(. \mid .)$$ the perplexity (of $$p_{\theta}$$ on $$X$$) is defined as follows:

<div id="eq">
\[
\text{ppl}(X) = \exp 
\left\{ 
-\frac{1}{t} \sum_{i=1}^{t} \log p_{\theta}(x_i \mid x_{< i}) 
\right\}
\]
</div>

* The quantity $$p_{\theta}(x_i \mid x_{< i})$$ represents the normalized score (i.e., probability) that the model generates the token $$x_i$$ after seeing the context $$ x_{\lt i} = (x_0, x_1, \ldots, x_{i-1})$$. 
* In practice, LMs usually outputs the logits (un-normalized scores), for each sequence input, we get a list of scores of size `vocab_size`. This is usually done in parallel over all input sequence.

## Code:

The following code snippet (provided by Hugging Face, see references section) computes the perplexity score of GPT2-small on Lambada. Hidden is the pre-processing code (model loading, tokenizer, and configs) followed by the main loop.

<details>
	<summary>Pre-processing: </summary>
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

* **Let's dissect the main loop:**

By matching the [equation above](#eq) and `ppl = torch.exp(torch.stack(nlls).mean())`. We understand that `nlls[i]` must represent the quantity:

\\[
\log p_{\theta}(x_i \mid x_{\lt i})
\\] 


* **What is the nature of:** `outputs = model(input_ids, labels=target_ids)`.


The variable `outputs` is of type `transformers.modeling_outputs.CausalLMOutputWithCrossAttentions`, and has three keys:

1. `outputs.loss`: a single scaler that apparently represnts the negative log likelihood loss of the current sequence.
2. `outputs.logits`: the output matrix of the LM, has a shape of `[1, seq_len, vocab_len]`.
3. `past_key_values`: will ignore for now.

* **How exactly can we compute the** `outputs.loss` **from the** `outputs.logits`:

The `output` matrix, compute the un-normalized scores of the next token of each input_token over the `vocab_size`. Hence, for each element in the sequence, you get a list of size `vocab-size` of un-normalized scores over... Also, the `target` is exactly a clone of the `input`. Hence, the first element of `target` shouldn't be used. But also the last element of `input` as we don't have it's ground truth at that moment, it would be computed at the next iteration of the loop. Hence, manually, this code snippet should do the job:


```python

# model outputs
logits, loss, _ = model(input_ids, labels=target_ids)

# softmax over dim 0 (over the vocab_sized for each input token)
logits_softmax  = torch.softmax(logits[0])

# for each input token, we gather the score assigned for it's true next token (`logits[input[ind], target[ind+1]` )



scores = torch.gather(logits_softmax, indices)[-1]

```


Test:

```python
# code


# for this:
print(f"Manual nll from logits >>  {}")
print(f"HF nnl output (output.loss) >> {outputs.loss}")
```

Hence, for each run:
* We are only interested in the model prediction for the **BEFORE** last token, which is **32002** in this example.
* We need to look at `outputs[0,-2]` and not `outputs[0,-1]`.
* `outputs[0, -2]` has a `[1, vocab_size]` shape. And, `outputs[0, -2][379]` would be represent exactly how much weight does the model think that the next token after 32002 would be **379**.
* `outputs[0, -2]` is not normalized. Hence, it should be softmax'd first. 

* **Important note** 

One should be careful as some models implements `input` and `target` differently. For instance, in Karpathy's GPT2 implementation, the `target` is usually `input[1:]` plus the true next token of `input[-1]`, where as Hugging Face models, expect `input` and `target` to be the exact same.

## Beyond perplexity:

Here I thought we discussed



#e References:

* [Hugging Face blog](https://huggingface.co/docs/transformers/en/perplexity)
* [The gradient blog](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)



