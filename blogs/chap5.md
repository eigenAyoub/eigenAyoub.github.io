---
layout: default
title:
permalink: /blogs/on-chip-memory/
---

**Content:**

* Arithmetic intensity.
* CUDA memory types.


## A primer on the arithmetic intensity:


Consider the main loop of a simple matrix  multiplication:

```cpp
for (int i = 0; k < width; k++){
	Pvalue += M[row*Width+k] * N[k*width + col]
}
```

The arithmetic (or computational) intensity is defined as the ratio of floating point operations **to** bytes accessed from the global memory (it's equal to 0.25 flops/byte in the example above).

An A100, DRAM peak bandwidth is 1555 GB/second, hence for the simple example above, we can barely do 389 GFLOPS (Giga Flops per second). This accounts to only 2% of the (theoretical) peak single-precision operations throughput ~ 19.5k GFLOPS. The peak operation throughput is equivalent to performing 12 ops per second.



## CUDA memory types:

![Memory](/src/media-gpu/mem.png) 


The Global memory (DRAM) and the Constant memory share the same access latency. The host can W and R on both, while the device can only R the constant memory. 

The Local memory is placed in Global memory, and has the same latency.
* BUT, it is not shared between threads, every thread has it's own region.
* Every thread places elements that can't be placed in their own registers:
	* Statically allocated arrays.
	* Spilled registers. 
	* Elements of the thread's call stack. (like what, wdym?)

Registers and shared memory are on-chip memories.
	* Registers are allocated to individual threads.
	* A kernel typically uses registers to allocate frequently (and privately) accessed variables to each thread. 
	* Shared memory is allocated to thread blocks, and shared between threads on the same block. Usually used to share data, and cooperate between threads, and share intermediate results. 



Differences between registers, shared memory and global memory:

* Global memory is implemented with DRAM technology (long access latency, low access bandwidth). 

* Registers: very short access latency, and drastically higher access bandwidth compared to the global memory.



Each access to registers involves fewer instructions than an access to the global memory. 

The scope and lifetime of each variable:



## Tiling:

Link to code: 



## Impact of memory usage on occupancy:





### Registers, how different are they used in CPU Vs GPU:

### The roofline model:

* The question: Is your program compute-bound or memory-bound.
* I am too dumb, always struggle to understand this.
