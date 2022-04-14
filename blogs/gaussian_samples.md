---
layout: default
title:
permalink: /blogs/gaussian_samples/
---

**Abstract:**

In this tutorial blog, we'll prove the correctness of the method animated above in a general setting. Elaborate on how we can use it to sample from the multi-variate Gaussian (MVN). Finally, we will discuss a practical approach and illustrate it with some code snippets from the Scipy and Numpy.

**Outline:** 

1. [Correctness: The transformation Law](#1)
   1. [Generating the exponential law from the uniform distribution](#11)
2. [MVN case: Normal x Normal = Normal](#2)
3. [Practical implementation:](#3)
   1. [Cholsky decomposition](#31)
   2. [Open Source](#32)



---

**Notation:**

If $$X$$ is a random variable (RV), we'll use $$F_X$$  to denote its commutative distribution function (CDF) and $$f_X$$ for its density function.

---
### **1. Correctness **<a name="1"></a>

Put in simple terms, if $$F_N^{-1}$$ is the reverse of the CDF $$F_N$$ of the standard normal distribution and $$U$$ is a uniformely distributed RV on the interval $$[0,1]$$. The animation above claims that $$ F_N^{-1}(U) $$ is normally distributed.

**Proof:**

Let's define the RV $$Y$$ as  $$Y = F_N^{-1}(U)$$  and let's prove that $$Y = N(0,1)$$ 

For $$y$$ in $$R$$ :

$$\begin{align*}
   F_Y(y) &= P(Y<y) \\
&= P(F_N^{-1}(U) < y) \\
&= P(U < F_N(y)) \\
&= F_N(y)
\end{align*}$$



Where the third equality follows from the fact that $$F_N^{-1}$$ is strictly increasing on the range $$[0,1]$$

Since the CDF completely determines a distribution, we conclude that $$Y$$ is equal to $$N$$, which completes our proof.



Note:  This actually is a simple version of the transformation law, i.e., when $$ Y = g(X)$$ s.t $$g$$ is strictly monotonic and differentiallable  on the range of $$X$$  , we can prove the following $$F_Y(y)  = f_X(g^{-1}(y)) \abs{\frac{d g^{-1} }{dy}(y)}  $$ 

In this case, $$X$$ is the uniform distribution and $$g$$ is the inverse CDF of the normal distribution.



#### **1.1. Example: The exponential law:** <a name="11"></a>
This example is very recurrent, mainly because of how important this distribution is and how we can analytically (and simply) get the CDF $$F_{Exp}$$ and its inverse $$F_{Exp}^{-1}$$ .

We have $$ F_X(x) = 1- e^{-\lambda x} $$  and   $$F_X^{-1}= $$ 

And following the same approach proved above, the RV $$ X = F_{Exp}^{-1}(U) $$ is definitely going to be  an exponential distribution.





---
### **2. Multi-variate case:**<a name="2"></a>

**Fancy Gaussian Features:**

The Gaussian distribution is known for many features. The "fanciest" ones to cite Including being the distribution that maximises the entropy for a given mean and variance, and namely how it weirdly appears in many asymptotic behaviours of the sums of i.i.d  RVs.

 **Practical reasons:**

The key reason why we keep using the normal distribution is its very convenient mathematical properties, including:

* An affine transformation of a Gaussian distribution is still Gaussian
* Product of many Gaussian is Gaussian
* Sub-sample of Gaussian is Gaussian 
* Conditional Gaussian are Gaussian ...

**Sampling from $$N(0_{R^{N}},I_N)$$  :**



The product property let us conclude that sampling $$N$$ one-dimensional  $$ X_i ~ N(0,1) $$ is actually th same as sampling  an $$N$$-dimensional RV $$ X = [X_1, .., X_N] ~ N(0_{R^{N}},I_N) $$ 

**Sampling from $$N(Nu,Sigma)$$ **



---
### **3. Practical implementation:** <a name="3"></a>

#### **3.1. The Cholseky decomposition:** <a name="31"></a>
One of the most famous decompositions in linear algbra is the Cholsky decompostion. In fact it is one of the most useful techniques in ML practice. Not only is it used for sam

The cholsky decompostion of a matrxi should be thought intuitively as the square root of a matrix

By the way, I'll devote a tutorial blog for the Cholsky decomposition and dive deeper in the theory and application of it. You can find it here when It's done: 

#### **3.2. Illustration from Scipy and Numpy:**<a name="32"></a>





---

**Some useful resources:**

* Some serious
* Philthinglip Hennig leklek erkejr
* Nando de Freitas course on

