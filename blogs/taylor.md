---
layout: default
title:
permalink: /blogs/taylor/
---


Problem setting and terminology:

We position ourselves in a classic supervised learning setting, and we are given:

* A finite training set $${(x_i, t_i)}$$ (drawn i.i.d from a $$p_{\text{data}}$$) .
* A loss function
* A class of parametric functions $${f_w(.)}$$


And we do so by minimizing the empirical risk over the training set:

\\[

\\]


Ideally though, we are interested in finding $$w$$ that minimizes the true risk:
\\[
R(w) = 
\\]


## First order Taylor Approximation:

Generally speaking:


In most cases in DL, $$f$$ is real valued, hence $$J$$ represents the gradient:

\\[

\\]

## Jax -- mini:

1. The gradient:

```python
def 
```

## Good questions:

1. Reverse mode autodiff, i.e., backprop (to compute gradients for instance in the case where f is a scalar function.


2. Forward mode autodiff:  compute the directional derivative

3. Computing with the Jacobian.


* How can the directional derivative determine how the network would change if the weights are slightly perturbed?

* How a model would change if we slightly perturb the training data?

* Hessian and inverse Hessian, check that section on Bishop

## The network's Jacobian:



