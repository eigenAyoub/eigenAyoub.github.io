---
layout: default
title:
permalink: /blogs/simd/
---


SIMD for GPUs.

**Content:**
* SIMD?
* Show me the instructions.

## SIMD

SIMD is a computer architecture class (following Flynn's taxonomy) that leverages the principle of a Single Instruction operating on Multiple Data. It is fundamentally a hardware design.SIMD processors are the basic paradigm of GPUS (although in GPUs it's more of SIMT). 

Assuming a SIMD device has multiple execution units (or processing elements), SIMD processing can either be in time or space. This time-space duality gives the following two sub-classes: 
* Array processors: The classic understanding of SIMD. 
	* Same instructions, on multiple data, on different PEs.
	* PEs can do different instructions in time.

* Vector processors:
	* Instruction operates on multiple data consecutively, using **the same PE**.
	* PEs do the same instruction over time.

![simd](/src/media-gpu/simd01.png)

For instance, in an A100 SM, the are PEs specific for loading/storing. 

In modern GPUs,


## PTX

SIMD can be internal (part of the hardware design) and it can be directly accessible through an instruction set architecture (ISA), but it should not be confused with an ISA.


PTX (Parallel Thread Execution) is NVIDIA 's low-level intermediate assembly language for CUDA. PTX serves as a virtual ISA that abstracts the underlying GPU architecture. When you compile CUDA code, it is first compiled to PTX code, before being compiled to the machine-specific code by the NVIDIA driver.



### **Things to add:**

* Memory banking?
* Registers in CPUs vs GPUs?
* Bottlenecks:
	* CPU-GPU transfer (PCIe, NVLink)
	* DRAM bandwidth.

**Title:**

SIMD, PTX, Memory banking issues, DRAM?
New architectures.
