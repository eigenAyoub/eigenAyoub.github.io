---
layout: default
title:
permalink: /blogs/cuda-softmax/
---


Make softmax go brr: The goal is incrementally speed-up my softmax in CUDA/C++. First, we only focus on an efficient implementation, then maybe worry about numerical instability.


Our baseline for GPU efficiency (and also correctness) is [CuDNN]().

### CuDNN:

Code to perform softmax:



### Tensor Cores.

tis was inspired by Sergey
