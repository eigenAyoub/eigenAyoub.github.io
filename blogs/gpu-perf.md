---
layout: default
title:
permalink: /blogs/gpu-perf/
---


Performance considerations. 

**Content:**
* 
*

The main bottlenecks in GPU computing are CPU-GPU data transfer, and global memory (DRAM) access, for each thread, DRAM access is usually 500x clocks cycles, registers access takes 1 clock cycle.  


## Memory access:

### Latency hiding and occupancy.

To hide latency, it is important to have enough warps in the SM Sub-Partition (SMSP). Occupancy is the ratio of active warps per max (can you even write english bro?).

Add an example of the occupancy calculator.
> use NSIGHT or .xls idgaf. Do it!


### Coalescing memory accesses:

How tf does a DRAM work?
![dram-top-down](/src/gpu-media/dram01.png)

* **Channels** connect the CPU/GPU to the memory itself.
* The memory itself is composed by DIMMs.
* Inside each **DIMM**, we have **ranks** which are composed of **chips** which contains memory **ebanks**.
* Banks are 2D-organized (rows/columns) memory cells.


Closer threads need to access data in the same regions. 

### Shared memory usage, avoid conflict related to banking/strides:


## Execution, SIMD and divergence.





Title:  Memory coalescing and thread coarsening.

## Memory coalescing:

DRAM cell:

Capacitor stores a charge, and three-state transistor based IC allowing data to be read/written.A

If the value stored is 1 >> it will discharge.


DRAM array:

Multiple DRAM cells, connected to a column wire. At a point of time, you can only read one DRAM cell of the array.


DRAM bank; is a 2D-array of DRAM cells // a stack of DRAM arrays, activated one row at a time, and red at the column.


How many elements can you fit in one array of a single DRAM bank?







































