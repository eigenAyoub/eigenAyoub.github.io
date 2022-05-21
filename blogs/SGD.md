---
layout: default
title:
permalink: /blogs/SGD/
---

Is this blog post, I will prove some results regarding the convergence of Stochastic Gradient Descent (SGD) in both non-convex and convex case, assuming an $$L$$-smooth $$\nabla f$$.

This was inspired by my personal work for the CMU course "Convex Optimization", taught by Ryan Tibshirani ([link to course](https://www.stat.cmu.edu/~ryantibs/convexopt-F16/)). Besides, the techniques presented here are pretty much textbook and present in every introductory course.

## Contents
* [Problem Setting](#problem-setting)
* [Non convex case](#nonconvex-case)
* [Convex case](#convex-case)

---
## Problem setting:

The goal is to minimize a differentiable function $$f$$ with
$$\mathrm{dom}(f)=\mathbb{R}^n$$, with an $$L$$-Lipschitz continuous gradient, i.e., for $$L>0$$:

$$
\| \nabla f(x) - \nabla f(y)\|_2 \leq L \|x - y\|_2, \;\;\; \text{for all $x,y$}.  
$$ 


Using the followig iterative procedure: starting from a point $$x^{(0)}$$, with each $$t_k \leq 1/L$$ : 

$$
x^{(k)} = x^{(k-1)} - t_k \cdot \nabla f(x^{(k-1)}), \;\;\;
k=1,2,3,\ldots,
$$

Generically written as follows: $$x^+ = x - t \nabla f(x)$$, where $$t \leq 1/L$$.


---
## Nonconvex case:

In this section, we will not assume that $$f$$ is convex. We will show that the gradient descent reaches a point $$x$$, such that $$\|\nabla f(x)\|_2 \leq \epsilon$$, in $$O(1/\epsilon^2)$$ iterations. 

>**Show that:** $$\, \, \,\,\, f(x^+) \leq f(x) - \Big(1-\frac{Lt}{2} \Big) t \|\nabla f(x)\|_2^2$$


<span style='font-size:20px;'>&#9654;</span> **Proof:**

At first, we prove this "101" property of $$L$$-lipschitz functions:

$$ \forall x,y \in \mathbb{R}^n \;\; f(y) \leq f(x) + \nabla f(x)^\intercal (y-x)  + \frac{1}{L} \|x-y\|^2 \;\;\;\;\;\;\textbf{(1)} $$

Let $$x,y$$ be in $$\mathbb{R}^n$$, Let's define the function $$g_{x,y}: \mathbb{R} \rightarrow \mathbb{R}$$ (we'll ommit the subscripts for simplicity) as $$g(t)=f(t x+(1-t) y)$$.
The function $$g$$ always appears in mysterious places, this is definitely one of them.

Some results regarding the function $$g$$:
* $$g(0)=f(y)$$ and $$g(1)=f(x)$$
* Using the chain rule we get: $$g'(t) = \nabla f(tx +(1-t)y) ^\intercal (x-y)$$
* Using the past line, we get: $$f(y)-f(x) = g(0) - g(1) = \int_1^0 \! g'(t) \mathrm{d}t $$

Building on the last property: 

$$
\begin{align*}
f(y)-f(x) &= \int_1^0 \! g'(t) \, \mathrm{d}t \\
&= \int_1^0 \! \nabla f(tx +(1-t)y) ^\intercal (x-y) \, \mathrm{d}t \\
&= \int_0^1 \! \nabla f(tx +(1-t)y) ^\intercal (y-x) \, \mathrm{d}t \\
&= \int_0^1 \! (\nabla f(tx +(1-t)y)-\nabla f(x)+ \nabla f(x))^\intercal (y-x) \, \mathrm{d}t \\
&= \int_0^1 \! \nabla f(x)^\intercal (y-x) \, \mathrm{d}t + \int_0^1 \! \nabla (f(tx +(1-t)y)-\nabla f(x))^\intercal (y-x) \, \mathrm{d}t\\
&= \nabla f(x)^\intercal (y-x) + \int_0^1 \! (\nabla f(tx +(1-t)y)-\nabla f(x))^\intercal (y-x) \, \mathrm{d}t \\
&\leq \nabla f(x)^\intercal (y-x) + \int_0^1 \! \|\nabla (f(tx +(1-t)y)-\nabla f(x))\| \| (y-x)\| \, \mathrm{d}t  \\
&\leq \nabla f(x)^\intercal (y-x) + \int_0^1 \! L \|tx +(1-t)y-x\| \| (y-x)\| \, \mathrm{d}t  \\
&= \nabla f(x)^\intercal (y-x) + L\| (y-x)\|^2\int_0^1 \! \vert t-1 \vert \, \mathrm{d}t  \\
&= \nabla f(x)^\intercal (y-x) + \frac{L}{2}\| (y-x)\|^2 \\
\end{align*}
$$

The first inequality is a direct application of Cauchy-Schwarz, and the following one, comes from the $$L$$-smoothness of $$\nabla f$$. This completes the proof of the inequality **(1)**.

Now, by plugging $$x^+ = x - t \nabla f(x)$$ in the placeholder $$y$$ and re-arranging, we complete our proof:

$$
\begin{align*}
f(x^+) &\leq f(x) + \nabla f(x)^\intercal (-t \nabla f(x))  + \frac{L}{2} \|-t \nabla f(x)\|^2 \\
&= f(x) - (1 - \frac{Lt}{2}) t \|\nabla f(x)\|^2 
\end{align*}
$$

<span style='font-size:32px;'>&#9632;</span>

>**Prove that:** $$\, \, \,\,\, \sum_{i=0}^k \|\nabla f(x^{(i)})\|_2^2 \leq \frac{2}{t} ( f(x^{(0)}) - f^\star)$$.


<span style='font-size:20px;'>&#9654;</span> **Proof:**

Using the definition of $$\{x^{(i)}\}$$ in the problem setting, and using the past result, we get for each $$i \in \{0, \ldots , k-1\}$$ : 

$$ \|\nabla f(x^{(i)})\|_2^2 \leq \frac{2}{t} ( f(x^{(i)}) - f(x^{(i+1)}) )$$

By summing each term, the RHS, cancels (Telescopes?), we get:

$$\, \, \,\,\, \sum_{i=0}^k \|\nabla f(x^{(i)})\|_2^2 \leq \frac{2}{t} ( f(x^{(0)}) - f(x^{(k)}))$$

Finally, we have (by definition), $$f^\star \leq f(x^{(k)})$$. We complete the proof (just upper bound the above mentioned inequality).

<span style='font-size:32px;'>&#9632;</span>

> Conclude that this lower bound holds: 

$$
\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|_2 
\leq \sqrt{\frac{2}{t(k+1)} (f(x^{(0)}) - f^\star)}, 
$$

<span style='font-size:20px;'>&#9654;</span> **Proof:**


We have for each $$i \in \{0, \ldots , k-1\}$$ 

$$\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|^2 \leq \|\nabla f(x^{(i)})\|^2 $$

Which implies 

$$
\begin{align}
(k+1) \min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|^2  &\leq \sum_{i=0}^k \|\nabla f(x^{(i)})\|_2^2 \\
&\leq \frac{2}{t} ( f(x^{(0)}) - f^\star)\\
\implies \\
\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|^2  &\leq \frac{2}{t (k+1)} (f(x^{(0)}) - f^\star) \\
\implies \\
\sqrt{\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|^2}  &\leq \sqrt{\frac{2}{t (k+1)} (f(x^{(0)}) - f^\star)} \\
\end{align}
$$

We complete the proof by simply verifying that:

$$
\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \| = \sqrt{\min_{i=0,\ldots,k} \|\nabla f(x^{(i)}) \|^2}  
$$


Which proves that we could achieve $$\epsilon$$-substationarity in $$O(1/\epsilon^2)$$ iterations.

<span style='font-size:32px;'>&#9632;</span>

---
## Convex case

Assuming now that $$f$$ is convex.  We prove that we can achieve $$\epsilon$$-optimality in $$O(1/\epsilon)$$ steps of SGD, i.e.,  $$f(x)-f^\star \leq \epsilon$$
 
> * Show that:
$$
\,\,\, f(x^+) \leq f^\star + \nabla f(x)^T (x-x^\star) -
\frac{t}{2}\|\nabla f(x)\|^2. 
$$

<span style='font-size:20px;'>&#9654;</span> **Proof:**

Using the first-order condition of convexity, we have $$ f(x) + \nabla f(x)^T (x^\star - x) \leq f^\star \,\, $$

which implies that $$f(x) \leq f^\star + \nabla f(x)^T (x - x^\star) \,\,$$ **(2)**

We have proved the follwing from the non-convex case:

$$\, \, \,\,\, f(x^+) \leq f(x) - \Big(1-\frac{Lt}{2} \Big) t \|\nabla f(x)\|_2^2$$

By re-arranging and then using the inequality **(2)**, we complete the proof:

$$
\begin{align}
f(x^+) &\leq f(x) - \Big(1-\frac{Lt}{2} \Big) t \|\nabla f(x)\|^2 \\
&\leq f(x) - \frac{t}{2} \|\nabla f(x)\|^2 \\
&\leq f^\star + \nabla f(x)^T (x-x^\star) - \frac{t}{2} \|\nabla f(x)\|^2 \\
\end{align}
$$

<span style='font-size:32px;'>&#9632;</span>


> Show the following:
$$
\,\,\, \sum_{i=1}^k ( f(x^{(i)}) - f^\star ) \leq
\frac{1}{2t} \|x^{(0)} - x^\star\|^2. 
$$

<span style='font-size:20px;'>&#9654;</span> **Proof:**

We first prove the following:

$$
 f(x^+) \leq f^\star + \frac{1}{2t} \big( \|x-x^\star\|^2 - \|x^+ - x^\star\|^2 \big). 
$$

Starting from the result we have proved in the past question, we use the generic update $$t \nabla f(x) = x - x^+$$, and a few arrangements to get the following:

$$
\begin{align*}
f(x^+) &\leq f^\star + \nabla f(x)^T (x-x^\star) -
\frac{t}{2}\|\nabla f(x)\|^2 \\
&= f^\star + \frac{1}{2t} ( 2t \nabla f(x)^T (x-x^\star) - t^2 \|\nabla f(x)\|^2 ) \\
&= f^\star + \frac{1}{2t} (2 (x-x^+)^\intercal (x-x^\star) - \|x - x^+\|^2 ) \\
&= f^\star + \frac{1}{2t} (\|x\|^2 - 2x^\intercal x^\star + 2(x^+)^\intercal x^\star - \|x^+\|^2 ) \\
&= f^\star + \frac{1}{2t} ((\|x\|^2 - 2x^\intercal x^\star + \|x^*\|^2) -( \|x^*\|^2 - 2(x^+)^\intercal x^\star + \|x^+\|^2 )) \\
&= f^\star + \frac{1}{2t} (\|x - x^*\|^2 - \|x^+ - x^*\|^2 ) \\
\end{align*}
$$


By summing the inequalities for each $$i \in \{0, \ldots , k-1\}$$, the RHS "telescopes", we are left with:

$$
\sum_{i=1}^k ( f(x^{(i)}) - f^\star ) \leq
\frac{1}{2t} (\|x^{(0)} - x^\star\|^2 - \|x^{(k)} - x^\star\|^2  )
$$

Since $$ 0 \leq \|x^{(k)} - x^\star\|^2 $$, we upper bound the RHS to complete the proof.


<span style='font-size:32px;'>&#9632;</span>

> Conclude with the following:
$$
\,\,\, f(x^{(k)}) - f^\star \leq \frac{\|x^{(0)} - x^\star\|^2}{2tk}, 
$$

<span style='font-size:20px;'>&#9654;</span> **Proof:**

This is pretty straightforward, every update from $$x^{(i)}$$ to $$x^{(i+1)}$$ makes the gap $$f(x^{(i)}) - f^\star$$ smaller, and hence, $$f(x^{(k)}) - f^\star = \min \{f(x^{(i)}) - f^\star\}_{i=1}^{i=k}$$, which justifies the following:

$$ 
\begin{align}
k(f(x^{(k)}) - f^\star) &\leq \sum_{i=1}^k ( f(x^{(i)})-f^\star ) \\
&\leq \frac{1}{2t} \|x^{(0)} - x^\star\|^2. 
\end{align}
$$

Second inequality follows from the past question. We complete the proof by deviding both sides by $$k$$.

<span style='font-size:32px;'>&#9632;</span>

This final result establishes the $$O(1/\epsilon)$$ rate for achieving $$\epsilon$$-suboptimality.  

## Summary:

In this blog post, we have prove the following:

* For a non-convex function $$f$$ with a Lipschitz $$\nabla f$$:
    * $$\epsilon$$-substationarity (i.e., $$\|\nabla f\| \leq \epsilon$$ ) is achieved in $$O(1/\epsilon^2)$$ steps

* For a convex function with a Lipschitz $$\nabla f$$:
    * $$\epsilon$$-optimality (AKA convergence) is achieved with $$O(1/\epsilon)$$ steps.



## TO-DO
* Proof checking
    * Proof checking 1 [Done]
    * Proof checking 2 [TO-DO]
    * Proof checking 3 [TO-DO]
* Convergence Rate for Proximal Gradient Descent [TO-DO]