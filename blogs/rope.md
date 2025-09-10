---
layout: default
title: 
permalink: /blogs/rope/
---

* **Paper:** RoPE (RoFormer: Enhanced Transformer with Rotary Position Embedding)
* **Link:** [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) 

---
### Setting:

Let $$\{\boldsymbol{x}_i\}_{i=1}^N$$, denote the embedding vectors (of d-dim) without positional information. Usually, the self-attention first incorporates position information to each $$\boldsymbol{q}, \boldsymbol{k}$$ and (optionally) $$\boldsymbol{v}$$, before computing the attention weights. Most methods ([review](https://arxiv.org/abs/2102.11090)) before RoPE **add** some* positional information to some** representation of the context.

* (*) Absolute or relative; learned or fixed.
* (**) Either the initial embedding vector, intermediate, or baked into the attention logits as a bias term.

For instance, for $$t\in\{q,k,v\}$$, a typical choice would be to add the positional encoding as follows: 

$$
f_{t}(\boldsymbol{x}_i, i) := \boldsymbol{W}_{t}(\boldsymbol{x}_i + \boldsymbol{p}_i)
$$

or to encode a positional bias into the attention logits in an additive (the usual case) or multiplicative (less usual) way as shown in [this paper](https://arxiv.org/pdf/2009.13658):

$$
e_{ij} = \frac{(x_i W^Q)(x_j W^K)^T a_{ij}}{\sqrt{d_k}}
$$

Where $$a_{ij}$$ is a learned scalar that encodes the relative position between $$i$$ and $$j$$.

Note: While this method uses a multiplicative term in the logits, it separates content from position. 

RoPE couples content with position via rotations at **multiple frequencies** (see expansion per block below for details). In one line: RoPE rotates the **queries** and **keys** in 2D blocks so the plain dot product becomes a function of relative offset.

---
## Details:

**For** $$d=2$$, RoPE does the following ($$\theta$$ is a constant):

$$
q_m=R(m\theta)\,W_q x_m,\quad k_n=R(n\theta)\,W_k x_n.
$$

with 

$$
R(\phi)=\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix},\quad  
J=\begin{pmatrix}0&-1\\[2pt]1&0\end{pmatrix},
$$

So the idea here is to rotate the key/query by an angle proportional to its position index. This implies incorporating the relative position in the dot product between the query and the key:


$$
\begin{aligned}
q_m^\top k_n
&= (W_q x_m)^\top R((n-m)\theta)\,(W_k x_n) \\[6pt]
&= x_m^\top W_q^\top\!\, R((n-m)\theta)\, W_k x_n.
\end{aligned}
$$

In the **general** case, let $$d$$ be the head dim (must be even), the $$q$$'s (and resp. $$k$$'s) are rotated as follows:

$$
q_m = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_q \boldsymbol{x}_m 
$$

where $$\boldsymbol{R}_{\Theta,m}^d$$ is the rotary matrix with pre-defined $$\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1:d/2]\}$$:

$$
\boldsymbol{R}_{\Theta,m}^d = \text{diag}(\boldsymbol{R}(m\theta_1), \boldsymbol{R}(m\theta_2), \dots, \boldsymbol{R}(m\theta_{d/2}))
\quad \text{s.t.:} \quad
\boldsymbol{R}(m\theta_i) = 
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
$$


Applying RoPE to self-attention, we obtain:

$$
\boldsymbol{q}_m^{\!\top}\boldsymbol{k}_n
= \hat{\boldsymbol{q}}_m^{\!\top}\,\boldsymbol{R}_{\Theta,n-m}^d\,\hat{\boldsymbol{k}}_n,
$$

with:

$$
\hat{\boldsymbol{q}}_m := \boldsymbol{W}_q \boldsymbol{x}_m,\qquad
\hat{\boldsymbol{k}}_n := \boldsymbol{W}_k \boldsymbol{x}_n.
$$

---
## Blockwise expansion:

Let the $$i$$-th 2D slices of $$\hat{\boldsymbol{q}}_m,\hat{\boldsymbol{k}}_n$$ be:

$$
\hat{\boldsymbol{q}}_m^{(i)}=\begin{pmatrix}q_{i,1}\\ q_{i,2}\end{pmatrix},\quad
\hat{\boldsymbol{k}}_n^{(i)}=\begin{pmatrix}k_{i,1}\\ k_{i,2}\end{pmatrix},
\qquad i=1,\dots,\tfrac{d}{2},
$$

Since

$$
\boldsymbol{R}(m\theta_i)^{\!\top}\boldsymbol{R}(n\theta_i)
=\boldsymbol{R}((n-m)\theta_i)
=\cos(\Delta\theta_i)\,\boldsymbol{I}_2+\sin(\Delta\theta_i)\,\boldsymbol{J},
$$

we obtain

$$
\hat{\boldsymbol{q}}_m^{\!\top}\boldsymbol{R}_{\Theta,\Delta}^d\hat{\boldsymbol{k}}_n
=\sum_{i=1}^{d/2}\!\Big(
\hat{\boldsymbol{q}}_m^{(i)}\!\cdot \hat{\boldsymbol{k}}_n^{(i)}
\cos(\Delta\theta_i)
\;+\;
\hat{\boldsymbol{q}}_m^{(i)\top}\!\boldsymbol{J}\,\hat{\boldsymbol{k}}_n^{(i)}
\sin(\Delta\theta_i)
\Big).
$$

**Non-separability:** 
The coefficients depend on $$q$$/$$k$$ and on block index $$i$$.
In general it cannot factor as
$$
\big(\hat{\boldsymbol{q}}_m^{\!\top}\hat{\boldsymbol{k}}_n\big)\,a(i,j).
$$

---
## How is it implemented? 



#### Hugging Face / Llama:

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

    #[...]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```



#### Mistral 7B

Obviously the French (math supremacy) will **complexicate** things!

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```


#### TODO:

* RoPE scaling in YaRN?
