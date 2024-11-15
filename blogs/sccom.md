---
layout: default
title:
permalink: /blogs/sensitive/
---

Title: Floating point arithmetic, from Ariane flight V88 to LLMs training.

## Motivation:

From, [Ariane flight V88](https://en.wikipedia.org/wiki/Ariane_flight_V88), To, LLMs training.

## Sources of error: 

Generally speaking, there two main sources of error induced by numerical approximations (ignoring errors induced by modeling assumptions):

1. Rounding: Computers can only do finite precision arithmetic.
2. Discretization: We approximate continues phenomenons with discrete counterparts.

While `Rounding` usually gets all the blame, errors stemming from discretization are as important if not even more.

**Example:** We want to approximate $$ \quad f'(x) \quad $$ using $$
\quad f_{\text{diff}}(x; h) \equiv \frac{f(x+h) - f(x)}{h} \quad
$$ for some small $$h$$. 

Assuming that $$f$$ is nice enough (twice diff and $$f''$$ is $$M$$-bounded) and using Taylor's theorem with a remainder, there exists some $$\theta \in [x , x+h]$$ :

$$
f(x+h) = f(x) + hf'(x) + f''(\theta) \frac{h^2}{2} 
$$

Which yields the following **discretization error**:

$$
f_{\text{diff}}(x; h) =   f'(x) + f''(\theta) \frac{h}{2}
\implies \left| f'(x) - f_{\text{diff}}(x; h) \right| \leq \frac{Mh}{2}.
$$


Additionally, we can introduce a **rounding error**. The numerator of $$f_{\text{diff}}$$ can not be computed exactly. Say we approximate it with $$\tilde{f}_{\text{diff}}(x; h) = \frac{f(x+h) - f(x) + \epsilon f(x)}{h} $$ with $$\epsilon \approx 10^{-16}$$. We get the following upper bound:

$$
\left| f'(x) - \tilde{f}_{\text{diff}}(x; h) \right|
\leq
\underbrace{\frac{Mh}{2}}_{\text{discretization err.}}
+ 
\underbrace{\frac{\epsilon f(x)}{h}}_{\text{rounding err.}}
$$


## Numerical sensitivity: 

Say $$y = f(x)$$, the idea is to measure how a small perturbation to the input $$x$$ reflects on the output $$y$$. Obviously, we want this rate of change to be small. For now, and w.l.o.g we define **the condition number** (of $$f$$)as follows:

$$C = \frac{\left|\Delta y  / y \right|}{\left|\Delta x  / x \right|}$$

* Condition number >> 1 

## Finite precision arithmetic:

Check the section from Matrix computation.

### Link to LLMs and host providers. 

* This is the interesting part
* Link to LLMs training, and hosting companies..
* Why training requires full precision, but hosting/inference is Ok?

## References:

* [Harvard Applied Math 205](https://people.math.wisc.edu/~chr/am205/)
* Matrix Computation. 
