---
layout: default
title:
permalink: /blogs/cuda-performance/
---

Based on lecture 06 and chapter 05.

Some obvious tricks that we've already discussed:

* Maximizing occupancy (playing with your `blockDim`, `gridDim`).
* Minimizing control divergence (high SIMD utilization).
* Shared memory usage and tiling.

This time:
* Memory coalescing.
* Thread coarsening.

## DRAM bank:

* How it works, on a high level:

![dram-bank](/src/media-gpu/dram/dram-bank-clear.png)

1. The incoming row address is decoded by the Row Decoder, which activates the corresponding row in the DRAM array.

2. The contents of that row are sensed and amplified by the Sense Amps, temporarily stored in Column Latches, and then passed through a multiplexer (MUX) where the specific columns—selected by the column address—are read out in a burst. 

![dram-bank](/src/media-gpu/dram/row-col-adr.png)

1. The second figure zooms in on the process of splitting the memory address: one portion selects the row while the other portion selects the columns within that row. 

2. After the appropriate row is activated, the data from the selected columns is routed through the MUX to be sent back on the data lines, facilitating a complete memory read or write transaction.

## DRAM bank:


## Memory coalescing:

* Memory coalescing is when threads within the same warp access elements from the same DRAM burst. 


Let's see a few examples of code:


``Cpp
int idx = blockDim.x*blockIdx.x + threadIdx.x
C[x] = A[idx] + B[idx]
```


