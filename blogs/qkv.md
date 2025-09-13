---
layout: default
title:
permalink: /blogs/KV-what/
---

This started with a question: **why is it called a $$\text{KV}$$ cache and not a $$\text{QKV}$$ cache?**  My (dumb) confusion came from mixing up **training** and **inference** modes.

- **Training.** We compute $$Q,K,V$$ for **all tokens** (per head x layer) and use them accordingly per head to compute the attention matrices. The whole sequence is processed in a single forward pass, hence why, we don't need to cache. 
- **Inference.** We generate tokens autoregressively (optionally after a prefill of the prompt). At each decode step $$t$$, there is an obvious opportunity to **reuse past $$K,V$$** from positions $$1..t$$, as the new query will attend over them. But my skill issues screamed at me: **don't you also reuse $$Q$$?**
- To make my point, I'll assume we cache everything, and I'll start from the final layer backwards, then discard every cache/intermediate state that we don't need along the way.

---

## Notation

- Decoder-only Transformer with $$L$$ layers, vocabulary size $$\lvert \mathcal V \rvert$$.
- $$d_{\text{model}}$$: model dim, $$d_h$$: head dim, and $$n_h$$: number of heads per layer.
- **Timeline:** we’ve produced $$t$$ tokens; and we're about to generate token $$x_{t+1}$$.
- Per layer $$\ell$$: queries $$Q^{(\ell)}$$, keys $$K^{(\ell)}$$, values $$V^{(\ell)}$$.

For the sake of the argument, assume we cached all three, per head x layer (to simplify notation, I will omit the head subscript from head specific matrices, unless if we're zooming in per head):

- $$Q^{(\ell)}_{1:t}\in\mathbb{R}^{t\times d_h}$$, $$K^{(\ell)}_{1:t}\in\mathbb{R}^{t\times d_h}$$, $$V^{(\ell)}_{1:t}\in\mathbb{R}^{t\times d_h}$$

Note: It is safe to discard all other operations (attention projection, ffn), as they all operate token-wise. For MHA, we can solely focus on per head logic, as detailed below. 

---
### Sampling $$x_{t+1}$$ from $$H^{(L)}_{1:t}\in\mathbb{R}^{t\times d_{\text{model}}}$$ :

Assume we are at step $$t$$ (we're about to generate token $$x_{t+1}$$). Obviously we only need the last hidden representation here $$H^{(L)}$$. 

To sample the next token, we only require the last row $$h_t^{(L)}$$, and it's usually done using unembedding $$W_U\in\mathbb{R}^{d_{\text{model}}\times \lvert\mathcal V\rvert}$$ (most likely the same as the embedding matrix, with weight tying):

$$
\text{logits}_{t+1} \;=\; h^{(L)}_t\, W_U \;\in\; \mathbb{R}^{1\times \lvert\mathcal V\rvert}, 
\qquad
x_{t+1}\ \sim\ \mathrm{softmax}(\text{logits}_{t+1}).
$$

In training, we pick the negative log likelihood of the correct token (true next token) for all positions (the labels are exactly the input shifted left with one step). A few snippets from [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master):

```python
def get_batch(split):
    # [...]
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # [...]
    return x, y
```

The forward pass of the model:

```python
    def forward(self, idx, targets=None):
        # [...]
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```


The **conclusion** here, is that to generate the next token, we only need to save the last row of the last hidden representation. Now we move backward, at each transformer layer, we know that we only need the last row of the output (of the layer), let's see what do need from the layer itself to compute the last row. 


---
## At layer $$\ell$$:

Let $$H^{(\ell)}_{1:t}\in\mathbb{R}^{t\times d_{\text{model}}}$$ be
the output of the **layer $$\ell$$**; we care **only** about the last row $$h^{(\ell)}_t\in\mathbb{R}^{1\times d_{\text{model}}}$$. So let's see what it depends on:

### **1) Strip position-wise parts (they don’t mix tokens)**

The tail of layer $$\ell$$ is position-wise (concat heads $$\to$$ $$W_O^{(\ell)}$$ $$\to$$ residual/norm $$\to$$ FFN). For the last row:

$$
h^{(\ell)}_t
=\underbrace{\mathrm{FFN}^{(\ell)}\!\big(\mathrm{LN}(h^{(\ell-1)}_t + o^{(\ell)}_t)\big)}_{\text{position-wise}}
+\,(h^{(\ell-1)}_t + o^{(\ell)}_t),
$$
with
$$
o^{(\ell)}_t \;=\; \big[z^{(\ell,1)}_t \,\|\, \cdots \,\|\, z^{(\ell,n_h)}_t\big]\, W_O^{(\ell)}.
$$

**Takeaway:** to get $$h^{(\ell)}_t$$ we need $$h^{(\ell-1)}_t$$ and the $$\{z^{(\ell,k)}_t\}_k$$. Hence, only the last rows are needed of $$H^{(l-1)}$$, and of each head $$Z^{(\ell,k)}$$.

### **2) Within a single head:**

For a single head $$k$$ of layer $$\ell$$ ($$k$$ to refer to a head is an ugly choice, but I've already referred to the activation per layer as $$h$$/$$H$$, and I'm too lazy now to change everything), the last-row of its output is $$z^{(\ell,k)}_t\in\mathbb{R}^{1\times d_h}$$, defined by:

$$
z^{(\ell,k)}_t \;=\; \alpha^{(\ell,k)}_{t,1:t}\, V^{(\ell,k)}_{1:t} \;\in\; \mathbb{R}^{1\times d_h}.
$$

$$
\alpha^{(\ell,k)}_{t,1:t} \;=\; \mathrm{softmax}\!\left(\frac{q^{(\ell,k)}_t \left(K^{(\ell,k)}_{1:t}\right)^\top}{\sqrt{d_h}}\right) \;\in\; \mathbb{R}^{1\times t}
$$

$$
q^{(\ell,k)}_t \;=\; h^{(\ell-1)}_t\, W_Q^{(\ell,k)} \;\in\; \mathbb{R}^{1\times d_h}
$$

**Conclusion:**

- To form $$z^{(\ell,k)}_t$$ you need **all values up to $$t$$**, $$V^{(\ell,k)}_{1:t}$$.
- To form $$\alpha^{(\ell,k)}_{t,1:t}$$ you need that **single** $$q^{(\ell,k)}_t$$ and **all keys up to $$t$$**, $$K^{(\ell,k)}_{1:t}$$.  
- To form $$q^{(\ell,k)}_t$$ you need **only** the **last input row** $$h^{(\ell-1)}_t$$ and $$W_Q^{(\ell,k)}$$.  



