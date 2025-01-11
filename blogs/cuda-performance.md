---
layout: default
title:
permalink: /blogs/cuda-performance/
---

**Optimizing performance of your CUDA code**

## Recap of the obvious:

* Latency hiding by maximizing occupancy (playing with your `blockDim`, `gridDim`).
	* But careful how much shared memory (register memory) you assign per block (resp. per thread), as it might lower your occupancy.
* Data reuse: Shared memory usage and tiling.
* Minimizing control divergence (high SIMD utilization).

## DRAM bank:

<div align="center">
	<img src="/src/media-gpu/dram/dram-bank-clear.png"  width="300">
	<img src="/src/media-gpu/dram/row-col-adr.png" width="300">
</div>

On a higher level, a DRAM bank (sketched above) does the following:

1. The incoming row address is decoded by the Row Decoder, which activates the corresponding row in the DRAM array.

2. The contents of that row are sensed and amplified by the Sense Amplifier, temporarily stored in Column Latches.

3. The Column Latches are then passed through a multiplexer (MAX) where the specific columns are selected based the Column address. 

The key factor here is that **if the next access** corresponds to **the same row address**, then we can the latency of Step 1 and Step 2 (which are the longest ones), and directly jump to Step 3, by fetching the necessary column from the Multiplexer. This is called **a memory coalesced access**.

## Memory coalescing on GPUs:


* Memory coalescing is when consecutive threads within the same warp access elements consecutive elements in the DRAM burst (hence, saving latency). 

* Again, always keep in mind how threads (within a block) are mapped to warps, and that `threadIdx.x` is the fastest moving dimension, followed by `threadIdx.y`, and then `threadIdx.z`:
	* For a 2D block, `tid = threadIdx.x + blockDim.x*threadIdx.y`. 
	* For a 3D block, `tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z`. 

Let's see a few examples of code:


```Cpp
int idx = blockDim.x*blockIdx.x + threadIdx.x
C[x] = A[idx] + B[idx]
```

## Credits:
* [Izzat El Hajj @ CMPS 224/396AA](https://ielhajj.github.io/courses.html)
* [Juan Gómez Luna @ 227-0085-51L](https://safari.ethz.ch/projects_and_seminars/spring2023/doku.php?id=heterogeneous_systems)
