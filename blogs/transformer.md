---
layout: default
title:
permalink: /blogs/transformer/
---

This is for now a Q&A based blog regarding the transformer architecture.

Jump to:
* [References](#ref)
* [Q: KV cache](#q2)
* [Q: Cross Attention, Masked Attention](#q7)
* [Q: How to compute Transformer weights?](#q4)
* [Q: Masking in BERT vs Masking in GPT](#q5)
* [Q: Flash Attention, Mistral 7B](#q6)
* [Q: Intuition behind Multi-head-attention](#q7)

TO-DO:

* Q: Multiple norm layers, why?
* Q: Training Vs Inferencing, differences?
* Q: If encoders-based model are good at understanding language, then why are decoder-only (e.g., GPT-family) models outperforming..
* Q: 

References 


---
<a name="q2"></a>**Question:** What's the KV cache:

It's the memory that is built up in the decoder during the generation (decoding) process, with every generated token $$t_i$$, its key (resp. value) $$k_i = x_i W_k$$, (resp. $$v_i = x_i W_v$$) gets appended to the right most column of $$K$$ matrix (resp.  bottom row of $$V$$.
 (typically context + prompt + generated tokens). 

This is for instance one of the key tricks to speed up training/inference. For instance, FlashAttention 

XXXX

This is a pretty advanced term. KV cache is the matrix KV that encoder builds as he decodes a se

At inference, at each time we sample a new toked, we are provided with a sequence that consists of a [promt + past  generated tokens].

At each step, the decoder requires self-attention of all these ast tokens, and hence, requires their KV.

What are universal transformers.

---
**Question:**  How Cross Attention Head (CAH) different form Multi-Head-Attention (MHA)?

In the CAH, we are fed the $$Q$$'s, and $$K$$'s from the decoder and $$V$$'s are fed the past MHA in the same attention layer. 



---
<a name="q4"></a>**Question:** The Transformer base archictecture has 65M parameters, how did we come up with it?

Let's first define the following terms:
* $$V$$: vocab size
* $$N$$: number of layers.
* $$h$$: number of heads per Multi-Head-Attention block.
* $$d_{model}$$: hidden size, aka, model dimension or embedding size.
* $$d_v, d_k, d_q$$: Output dimension of the Weight Matrices ($$W_V, W_K, W_Q$$).
	* We'll assume that $$hd_v = h d_k = h d_q = d_{model}$$ (A practice that is largely followed in most papers/implementations)
* $$W_o$$: This is the matrix that is multiplied after the contacatenation of the heads, has size $$d_{model}^2$$



The general formula could be derived as follows, I'll ignore the biases and normalizartion params: 

$$T = E +  N \underbrace{(  \overbrace{8 A + d_{model}^2}^\text{Multi-head-attention} + F)}_\text{1 Enc. Block} + N( \underbrace{2(8 A + d_{model}^2) + F}_\text{1 Dec. Block}) $$

$$F$$ is the feed forward layer, historically, It has always been assumed tobe a one hidden layer, of $$4 d_{model}$$ size. $$\rightarrow F  = 4 d_{model} d_{model} + $$

Plugging the numbers  for Transfor based ($$N=6, h=12, d_{model} = 512, d_v = 64$$) would lead to $$T = 6xM$$

For the larger model ($$N=6, h=12, d_{model} = 512, d_v = 64$$) I get $$T = 114M$$ which X over the paper numbers...

A few notes:

* The embedding matrix $$E$$ represents X% of the weights of the base $$65M$$ Transformer model. Let that sink in!

---
<a name="q6"></a>**Question:** What's so special about Mistral 7B.

Mistral 7B is a model that came up recently (Oct 2023). It has a few new ways

---
<a name="q7"></a>**Question:** What are some of the drawbacks of single-headed attention?

Some vectors could take over. I.e., $$q^T k_i$$ could way larger for certain key $i$. And hence, get more (undeserved?) attention. 



Let's consider three sets of vectors, Values, Queries and Keys:  $\{v_1, \dots , v_n \} \sub R^d$

$\{q_1, \dots , q_n \} \sub R^d$, and $\{k_1, \dots , k_n \} \sub R^d$ 



It was first used in it's modern context in the paper >> Check Karpathy video. An LSTM also uses a similar attention mechanism. 




---
<a name="q5"></a>**Question:** What is masking? How is it different in BERT vs GPT-based models.

* Masking in a decoder mean one thing: you can't the future tokens in the cross attention  layer.
* Usually implemeted using this underrated trick:

* Masking in BERT is more general, in fact it is the heart of BERT (hence why BERT is called an MLM, i.e., masked language model).   

Masking in decoders is a trick to not cheat while training the model. During training, we usually feed a whole sequence into the model, a train the model to predict the next token. Let's say the we have a training sequence of the following length $$(x_1, \dots, x_L)$$

Masking in decoders is a way to prevents the present token of accessing information about the future tokens.

What is usually done during training is to map this single coherent sentence to multiple training inputs, like the following:

* $$(x_1)$$  $$\rightarrow$$ We "wish" the decoder could generate $$x_2$$
* $$(x_1,x_2)$$  $$\rightarrow$$ We "wish" the decoder could generate $$x_3$$
* $$(x_1, \dots, x_{L-1})$$ $$\rightarrow$$ We "wish" the decoder could generate $$x_L$$
* $$(x_1, \dots, x_{L})$$ $$\rightarrow$$ We "wish" the decoder could generate  <EOS> token (End Of Sequence).


Masking in BERT is what makes BERT... BERT. BERT is *NOT* pre-trained to generate the next token (i.e., it not an auto-regressive model). It's a MLM (Masked Language Model).


Worth mentioning that BERT is also trained using a NSP (next sentence prediction), but it was shown later on that is almost obsolete, longer MLM training with longer sequence was enough (Check [RoBERTa](https://arxiv.org/abs/1907.11692))

---
### <a name="ref"></a> References:

* Original Transformer Paper [[Link](https://arxiv.org/abs/1706.03762)]
* BERT [[Link](https://arxiv.org/abs/1810.04805)]
* RoBERTa
* Kipply's great blog on ..
