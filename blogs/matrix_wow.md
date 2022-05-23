---
layout: default
title:
permalink: /blogs/enter_the_matrix/
---

**Abstract**

Maintaining a healthy relationship with matrices is a crucial asset in modern ML practices. In this blog I'll collect some of the most known transformations/operations that I have encountered and used. Perhaps make it a "bigger" project, as I'm currenlty learning JAX. We'll see!

## Content:
1. [The Passepartout](#the-passpartout)
2. [References](#references)




## The Passepartout:


1.  $$\frac{\partial b^\intercal a}{\partial a} = b$$

2.  $$\frac{\partial a^\intercal A a}{\partial a} = (A^\intercal + A)a$$

3.  $$\frac{\partial \log (\det A) }{\partial A}= (A^{-1})^\intercal$$

4.  $$\frac{\partial tr(BA)}{\partial A}= B^\intercal $$

5. $$\frac{\partial A^{-1}}{\partial \alpha} = - A^{-1} \frac{\partial A}{\partial \alpha} A^{-1}$$

6. $$\frac{\partial \det A }{\partial A}= (\det A) A^{-1} \;\;\;\; \text{Such a beauty!}$$  





## References:

* Stefan Harmeling, ML course, lecture 08/28. [Link](https://www.youtube.com/watch?v=uQ8Q9B1LMVw)


&nbsp;
&nbsp;

---

**Progress:**

* First skeleton [Done]
* Polish the latex (scalars / vectors / matrices) when it's not clear from the context [TO-DO]
* Keep adding formulas.  [TO-DO]
* Add the generic procedure to compute complicated forms (Differential / Identification rules) [TO-DO]
* Perhaps add some JAX and make it a worthy bog-post [Really TO-DO]



