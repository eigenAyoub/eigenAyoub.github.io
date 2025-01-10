---
layout: default
title:
permalink: /blogs/on-chip-memory/
---

**Content:**

* [Occupancy of an SM.]()
* [Arithmetic intensity.]()
* [CUDA memory types.]()

## Occupancy of an SM:

First know the specs of your GPU (e.g., an A100, [Code](https://github.com/eigenAyoub/cuda-linear-alg/props.cu)):

```bash
cc $ nvcc -o props props.cu  && ./props
Number of CUDA devices: 1
Device 0: NVIDIA A100 80GB PCIe
Compute capability: 8.0
Total global memory: 79.1384 GB
Shared memory per block: 48 KB
Shared memory per SM: 164 KB
Registers per block: 65536
Warp size: 32
Max threads per block: 1024
Max threads per SM: 2048
Max threads dimensions: 1024 x 1024 x 64
Max grid size: 2147483647 x 65535 x 65535
Clock rate: 1410000
Total constant memory: 65536
Texture alignment: 512
Multiprocessor count (#SMs): 108
```

The occupancy per SM is defined as follows: $$\frac{\# \text{active warps per SM}}{\text{max # of warps per SM}}$$.

For instance, if I choose a block size of 32 threads, I would have 1024 threads per SM (32 blocks per SM, each has 32 threads), which yields an Occupancy of 1/2, as the maximum number of threads for an A100 is 2048 (64 warps per SM).

**Notice** how the Shared memory per SM `IS NOT EQUAL` to [{shared memory per Block} x {#max number of blocks per SM}]. If your block is using too much shared memory, then it would limit the number of blocks assigned to an SM.

## Performance bounds and arithmetic intensity:

**Bounds or bottlenecks:**

The question you ask: Why can't your program can't run faster? Two answers:
1. If only my GPU can do more operations/second -> **Compute bound**.
	* Data transfer is not an issue. There is just not enough compute.
2. If only my GPU can move more data/second -> **Memory bound**.
	* Some compute cores might be idle, waiting for data.

**Arithmetic intensity:**

The arithmetic (or computational) intensity is defined as the ratio of floating point operations **to** bytes accessed from the global memory. For instance, it's equal to 0.25 flops/byte in the following example.

```cpp
for (int i = 0; k < width; k++){
	Pvalue += M[row*Width+k] * N[k*width + col]
}
```

An A100, DRAM peak bandwidth is 1555 GB/second, hence for the simple example above, we can barely do 389 GFLOPS (Giga Flops per second). This accounts to only 2% of the (theoretical) peak single-precision operations throughput ~ 19.5k GFLOPS. The peak operation throughput is equivalent to performing 12 ops per second.


**Desired compute intensity.** 

Consider we have the following specs:

* **GPU FLOPs**: 14 TFLOP/s (14,000 GFLOP/s)  
* **Memory Bandwidth**: 900 GB/s  

A desired compute-to-memory-access ratio is: 

$$\frac{14{,}000 \,\text{GFLOPS}}{900 \,\text{GB/s}} \approx 15.6 \,\text{FLOP/byte}.$$

<span>&#9888;</span> **Important >** Each single-precision floating-point operation (FLOP) operates on **4 bytes** (32 bits) of data. Thus:

$$
\text{FLOP/byte} \times 4 \rightarrow 15.6 \times 4 \approx 62 \text{Per Floating point access}
$$



<!--## The roofline model:

* The question: Is your program compute-bound or memory-bound.
* I am too dumb, always struggle to understand this.

## Tiling, and how it improves the arithmetic intensity:

* Naive matrix multiplication has a ration of `0.25 FP / B` (embarassing).



## CUDA memory types:  // move this section to a separate file.

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
### Registers, how different are they used in CPU Vs GPU:
-->
