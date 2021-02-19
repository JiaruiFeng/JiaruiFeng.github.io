---
title: "Kernel Function"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide


---



### Introduction

First, we consider 3-rd order polynomial function:$y=\theta_{3}x^3+\theta_{2}x^2+\theta_{1}x+\theta_0x$. Views it as a non-linear transformation $\phi(x)=[1,x,x^2,x^3]^T$, $y=\theta\phi(x)^T$, where $\theta=[\theta_0,\theta_1,\theta_2,\theta_3]$. We can see that we only have one original variable, after non-linear transformation, it become three dimension variable. Usually, the computation is extremely expensive when $\phi(x)$ is  high-dimensional. However, we can use kernel trick to avoid such a expensive computation.

#### Kernel Trick

Consider gradient descent in Linear regression:


$$
\theta:=\theta+\alpha\sum_{i=1}^{n}\left(y^{(i)}-\theta^{T} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right)
$$


Initialize $\theta=0 $ and let $\theta=\sum^{n}_{i=1}\beta_i\phi(x^{(i)})$:


$$
\theta:=\theta+\alpha \sum_{i=1}^{n}\left(y^{(i)}-\theta^{T} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right)=\sum_{i=1}^{n} \underbrace{\left(\beta_{i}+\alpha\left(y^{(i )}-\theta^{T} \phi\left(x^{(i)}\right)\right)\right)}_{\text {new } \beta i} \phi\left(x^{(i )}\right)
$$


We can notice the update rule for $\beta_i$ is:


$$
\begin{aligned}
\beta_{i}:&=\beta_{i}+\alpha\left(y^{(i)}-\theta^{T} \phi\left(x^{(i)}\right)\right)\\
&=\beta_i +\alpha(y^{(i)}-\sum^{n}_{j=1}\beta_j\phi(x^{(j)}) \phi(x^{(i)}))\\
&=\beta_i +\alpha(y^{(i)}-\sum^{n}_{j=1}\beta_j\mathcal{<}\phi(x^{(j)}),\phi(x^{(i)})\mathcal{>})
\end{aligned}
$$


we can update $\beta_i$ every time to get finally $\theta$. But there are several properties for this formula. First, notice that $\mathcal{<}\phi(x^{(j)}),\phi(x^{(i)})\mathcal{>}$ wouldn't change for every update step. Thus we can compute it before we do gradient descent. Second, $\mathcal{<}\phi(x^{(j)}),\phi(x^{(i)})\mathcal{>}$ can be efficient for specific $\phi(x)$, and these kind of specific $\phi(x)$ we call kernel function.

#### Kernel Function

We define Kernel corresponding to feature map $\phi$ as a function that maps $\mathcal{X}:\mathcal{X}\rightarrow\mathbb{R}$ satisfy :


$$
K(x, z) \triangleq\mathcal{<}\phi(x), \phi(z)\mathcal{>}
$$


This way, our algorithm can become:


$$
\beta:=\beta+\alpha(Y-K\beta)
$$


Notice that the complexity for this is $O(n)$. For the prediction, we can write:


$$
\theta^T\phi(x)=\sum^{n}_{i=1}\beta_i\phi(x^{(i)})^T\phi(x)=\sum^{n}_{i=1}K(x^{(i)},x)
$$


Some properties for kernel function:

1. This is one kind of non-linear transformation.
2. We reduce the computation from $O(n^2)$ to $O(n)$
3. Kernel function is also a similarity metrics: $K(x,z)=\phi(x)^T\phi(z)$. If $x$ and $z$ is similar, $K$ would be large.

### 2. Necessary Condition for Valid Kernels

We define kernel matrix as $K$, where $K_{ij}=K(x^{(i)},x^{(j)})$ for $\{x^{(1)},x^{(2)},...,x^{(n)}\}$. the necessary and sufficient condition is: **$K$ is semi-definite and symmetric**.

Now, let's use some common kernel function.

#### Polynomial kernel

This kernel is popular in image processing. The formula is:


$$
K\left(x,z \right)=(xz+1)^d
$$


Where $d$ is a hyper-parameter.

#### Radial basis function kernel（RBF)

RBF is one of the famous kernel function we use in machine learning.The formula is follow:


$$
K\left(x,z \right)=\exp \left(-\frac{\left\|x-z\right\|_{2}^{2}}{2 \sigma^{2}}\right)
$$


This function can project the original data to a infinite dimension. To see this, we can use Taylor expansion. Set $\sigma=1$ for simplicity:


$$
\begin{aligned}
\exp \left(-\frac{\|x-z\|^{2}_2}{2}\right) &=\exp \left(-\frac{(x-z)^{T}(x-z)}{2}\right) \\
&=\exp \left(-\frac{x^{T} x-2 x^{T} z+z^{T} z}{2}\right) \\
&=\exp \left(-\frac{\|x\|^{2}}{2}\right) \cdot \exp \left(-\frac{\|z\|^{2}}{2}\right) \cdot \exp \left(x^{T} z\right) \\
&=\sum_{n=0}^{\infty} \exp \left(-\frac{\|x\|^{2}}{2}\right) \cdot \exp \left(-\frac{\|z\|^{2}}{2}\right) \cdot \frac{\left(x^{T} z\right)^{n}}{n !}
\end{aligned}
$$


We can see the third term is actually $\infty$-order polynomial function time a constant. That's why RBF can project original data to infinity dimension.



### Reference

[1] CS229: Machine Learning, Stanford. http://cs229.stanford.edu 