---
layout: default
title:
permalink: /blogs/cuda-indexing/
---


This footnote from PMPP bothered me a lot (page 51).

![whywhywhy](/src/media-gpu/whywhywhy.png)

Like Why? Why complicate? Ok, some clarifications:

**Grid and Block dimensions:**

You always define `gridDim` and `blockDim` in the (x, y, z) order that CUDA expects.
[Add an example here for an input of (256, 128), `gridDim.y` should equal `256/blockDim.y`]

**Mapping to your problem:**

Inside the kernel, you choose how to interpret `threadIdx.x`, `threadIdx.y`, and `threadIdx.z`. For a 2D array, a typical choice is:

```Cpp
// Map x -> columns, y -> rows
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```
This tends to match row-major order, where columns (x) are the fastest-changing index in memory. And within a CUDA block, `threadIdx.x` is the fastest changing one, then It would better (for your mental health) to visualize the data within the kernel as (z,y,x). 


**The confusion:**  lays in how in math/C++, the column dimension is usually mapped to the axis `y`. And it is the fastest changing one. For you mental health, forget about this. In CUDA programming, the x-axis is the last one (as it is the fastest changing), but we still declare `gridDim` and `blockDim` is the usual `(x,y,z)` order. But, it would be helpful -- for instance for a 2D data, to visualize the y-axis as the vertical one, and the x-axis as the horizontal one.

---
### What is a  Coalesced Memory Access?

* Coalesced memory access is (a pattern?, a property?) when consecutive threads in a warp access consecutive memory locations. E.g., if threads in a warp access memory locations like `A[0], A[1], A[2], ... A[31]`, the memory access is coalesced. If threads access scattered locations like `A[0], A[100], A[200], ...`, the access is non-coalesced and inefficient.



* A CUDA block is split into **warps** of 32 threads, where `threadIdx.x` is the fastest-varying dimension. Hence, aligning `threadIdx.x` with the fastest-varying dimension of your data layout promotes coalesced access.

---
### A toy example.

Consider a 2D array stored in **row-major order**, where consecutive elements of a row are stored contiguously in memory:

```plaintext
Matrix A (3x4):
[
 [a00, a01, a02, a03],
 [a10, a11, a12, a13],
 [a20, a21, a22, a23]
]

Linear memory layout:
[a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
```
---
### Coalesced Memory Access Kernel


Consider the following kernel:

```Cpp
#include <stdio.h>

__global__
void kernelCoalesced(float* d_data, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        int index = row * width + col;
        d_data[index] += 1.0f;
    }
}
```

**Explanation:**

* Consecutive threads (`threadIdx.x`) move to consecutive columns in the same row, which aligns with how data is laid out in memory, which promotes a coalesced memory access.

---
### Non-Coalesced Access Example

Now consider a kernel that accesses the **same** 2D array in **column-major order** (think of it as if we needed to operate on the `transpose` of a matrix that is originally laid-out in a row-major order):

```Cpp
__global__
void kernelNonCoalesced(float* d_data, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width)
    {
        int index = col * height + row;
        d_data[index] += 1.0f;
    }
}
```

**Explanation:**
- thread 0 > col = 0, row = 0, hence, it would access `d_data[0]`
- thread 1 > col = 1, row = 0, hence, it would access `d_data[height]`
- This layout, indices map does not generally promote coalesced memory access.




