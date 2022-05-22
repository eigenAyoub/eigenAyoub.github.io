---
layout: post
title:
permalink: /blogs/gaussian_samples/
---

**Abstract:**

In this tutorial blog, we'll prove the correctness of the method animated above in a general setting. Elaborate on how we can use it to sample from the multi-variate Gaussian (MVN). Finally, we will discuss a practical approach and illustrate it with some code snippets from the Scipy and Numpy.

**Outline:** 

1. [Correctness: The transformation Law](#1)
   1. [Generating the exponential law from the uniform [TO-DO]distribution](#11)
2. [MVN case: Normal x Normal = Normal](#2) [TO-DO]
3. [Practical implementation:](#3) [TO-DO]
   1. [Cholsky decomposition](#31)




---

**Notation:**

If $$X$$ is a random variable (RV), we'll denote its CDF (and resp. density function) by $$F_X$$ (resp. $$f_X$$).

**Idea:**

Let $$F_\mathcal{N}^{-1}$$ be the reverse of $$F_\mathcal{N}$$, the CDF of the standard normal distribution and $$\mathcal{U}$$ be a uniformely distributed RV on the interval $$[0,1]$$. The animation above claims that $$F_\mathcal{N}^{-1}(\mathcal{U})$$ is normally distributed.

**Proof:**

Let's define the RV $$Y$$ as  $$Y = F_\mathcal{N}^{-1}(\mathcal{U})$$   and let's prove that $$Y = \mathcal{N}[0,1]$$ by proving that $$F_\mathcal{N} = F_Y$$

For $$y \in \mathbb{R}$$ :

$$\begin{align*}
F_Y(y) &= P(Y \leq y) \\
&= P(F_\mathcal{N}^{-1}(\mathcal{U}) \leq y) \\
&= P(\mathcal{U} \leq F_\mathcal{N}(y)) \\
&= F_\mathcal{N}(y)
\end{align*}$$



Where the third equality follows from the fact that $$F_\mathcal{N}^{-1}$$ is strictly increasing on $$[0,1]$$

Since the CDF completely determines a distribution, we conclude that $$Y$$ is equal to $$\mathcal{N}$$, which completes our proof.


**To-Do:**
* Check the [To-Do]'s
* Complete the blog post, make it interesting!!! (Before June 2022)