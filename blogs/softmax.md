- **Paper:** Scalable-Softmax Is Superior for Attention
- **Link:** [https://arxiv.org/abs/2501.19399](https://arxiv.org/abs/2501.19399)
- **Why read:** Was referenced in Lamma 4
> Additionally, we employ inference time temperature scaling of attention to enhance length generalization

### Map:

* [Motivation](#motivation)
* [Details](#details)
* [How](#how)

### Motivation:

* Paper enhances softmax within attention blocks as a mean to improve context length generalization.
* Solves the `Attention fading` problem: as the size of the input grows, softmax scores flatten, hence, poorly detects important tokens.

### Details:

For an input vector $$z$$ of size $$n$$, Softmax is defined as follows:

$$
z_i \mapsto \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} 
$$

The proposed Scalable Softmax (SSMax):

$$
z_i \mapsto \frac{n^{s z_i}}{\sum_{j=1}^{n} n^{s z_j}} = \frac{e^{(s \log n) z_i}}{\sum_{j=1}^{n} e^{(s \log n) z_j}},
$$

with $$s$$ being "the" scalar parameter (a learned one per head/layer).

* In the new formulation of SSMax, the exponential depends on $$n$$ as well, i.e., even if the input size grows, the nominator also follows.

### How?

Experimentally set the softmax for all layers/heads for each input size $$n$$ as:

$$
z_i \mapsto \frac{e^{(sp_n+b)z_i}}{\sum_{j=1}^{n} e^{(sp_n+b)z_j}},
$$

with $$s$$, $$b$$ being learnable parameters per layer/head, and $${p_n}$$ learnable but shared between all layers/heads, and depending only on context size $$N$$. 

Experimentally they noticed that $$p_n$$ follows a logarithmic scale: $$p_n \approx a_n \log(n) + b_n$$. They decided to drop the $$\{b_n\}_n$$ and only learn the $$\{a_n\}_n$$ per head/layer.

