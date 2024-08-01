---
layout: default
title:
permalink: /blogs/llama/
---


Some personal notes from Llama 3.1 technical report >  

Links:

https://ai.meta.com/research/publications/the-llama-3-herd-of-models/ 
https://www.reddit.com/r/LocalLLaMA/comments/1eabf4l/lets_discuss_llama31_paper_a_lot_of_details_on/



## Main take away-s: 

* Dense transformer model (nothing fancy, no MoE). 
* 408B parameters, 128K context window.
* 15T tokens.
* Image, video and audio are added via a compositional approach for test purposes only, not yet prod ready?

* Keys: Data, Scale, and managing complexity.
	* While data and scale are obvious terms.
	* Managing complexity, i.e., we have enough compute to just keep simple and scale. Less fancy architecture, more engineering `flex`.


* What has been released: pre-training, post-training and Llama Guard model (in/output safety?)


## General overview:

* Pre-training: Massive training of the 408B model on 15.6T tokens. First a standard pre-training on 8k context window, and then followed by continued pre-training on 128K context window.

* Post-training: Align the model with HF, each round with SFT on instruction tuning data and DPO (link to DPO). Utilities added in post-training: tool use, safety measures.



## Pre-training:

### Data cleaning:

Without saying much, clearly of engineering has been spent to make such high quality data. 

Data mix:
* Annealing Data: wtf is this?  3.4.3

### General architecture:

All notes there are equally important.

### Infra stuff:

This is just nuts.

MAST
Tectonic
* tf does this mean: we aim to minimize GPU pause time during checkpointing and increase heckpoint frequency to reduce the amount of lost work after a recovery.
