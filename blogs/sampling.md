---
layout: default
title:
permalink: /blogs/sampling/
---


Exact inference algorithms are mainly split into two categories, variational methods and sampling based methods. In this blog, I'll dive into the latter (and OG) part. I'll be mainly using Bishop (Chapter 11) and Mackay (Chapter 29), and try to focus more on the intuitive side.

## Content:
1. [Sampling: An introduction](#sampling-an-introduction)
2. [Simple techniques](#simple-techniques)
3. [Monte Carlo](#monte-carlo)
4. [Monte Carlo Markov Chain](#monte-carlo-markov-chain)
5. [Miscellaneous](#miscellaneous)
6. [References](#references)


## Sampling: An introduction

Sampling and simulation based methods are key to many scientific advances that occured in the 20th century, including some unfortunate events ([History](https://en.wikipedia.org/wiki/Monte_Carlo_method#History), [Manhatten Project](https://en.wikipedia.org/wiki/Manhattan_Project)). Let's jump to the heart of it.

Without a doubt, the most used sentences in every bayesian stats textbook or paper are:
<div style="text-align: center;">
  It's good BUT it's <b>intractable</b> and ineficient to compute the evidence. <br>
  The <b>integral</b>, is... hard to compute.  <br>
</div> 

Fortunetely, MC methods are one possbile way out.

**Problem setting and key idea:**


$ P = $

We want:
* Genrate samples *$$\{x_{r}\}$$*
	* Usual problem: We don't have access to P, but we can evaluate it efficiently.
	* Why is it relevant:
		* Aspect 1
		* Aspect 2
		* Aspect 3
* Compute the $$ E = 1 $$,  usually in ML, It is a nasty (Like how Mackay used to say) Posterior distibution. 
	* For instance, the famous DQN objective in Reifocement Learning:
		* $$ E $$ , from the famous Mnih et al




**Why is sampling hard?**

**Correctness: Law of Large numbers**


## Simple techniques

* Transformation of variables:


* Inverse CDF


* Fancy transformations: Normalizing Flows


## Monte Carlo

Rejection sampling and Importance sampling


## Monte Carlo Markov Chain

Markov Chain:

Ergodocity: 

Run it long enough, you'll eventually get a point that comes from P ?

Theorem:
if X_0, X_1, .. in an irreducible, time-homogeneous, discret Markov Chain with a stationary distribution pi, then:

In other words, we could use the state that the Markov Chain converges as a sample of pi! Let it sink in.






## Misc

Application:
* RL methods?
* Inverse methods?

## References

* Mackay (), Chapter 29
* Bishop (2006), Chapter 6
* Normalizing Flows

* Mackay course on Youtube, Lecture 12 and 13
* Phillip Hennig course, Lecture 3 and 4
