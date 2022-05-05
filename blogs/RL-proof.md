---
layout: default
title:
permalink: /blogs/TD-learning/
---


# Proof of convergence of TD-Learning:
$$
\nabla_\phi \mathbb{E}_{q(\mathbf{z}|\phi)}\left[ f(\mathbf{z}) \right] ~=~ 
\mathbb{E}_{q(\mathbf{z}|\phi)}\,[
   f(\mathbf{z})\, \nabla_{q(\mathbf{z}|\phi)} \log q(\mathbf{z}|\phi) ]
\hspace{41mm}(1)
$$


In this short blogpost, I'll present a proof of the famous famous RL algirthm, TD-learning [1], in finite state-action MDPs. This was part of the Homework #2 of the Toronto course CSC2547 ("Introduction to RL" taught by Prof. Amir Massoud). [Link to my github repo]()

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$


## Content:
* [Setting](#setting)
* [Algorithm](#algorithm)
* [Proof](#proof)
	* [Sketch](#sketch)
	* [Proof](#proof)
* [References](#references)


## Setting:
We have an MDPs $$ $$


## Algorithm:

## Proof:
$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$
$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$

### Sketch:
The main idea is to ....
$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$

$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$$$
\begin{aligned}
V(s) &= \mathbb{E}[G_t \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\
&= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s]
\end{aligned}
$$


### Proof:
Let this and that be ...


## References:
* [] Sutton & Barto
* [] Link to TD-learnig paper
* [] Amir massoud Fahramad
* [] Link to youtube playlist, github linl

