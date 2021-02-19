---
title: "Regularization"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide
---



Regularization is the first tool help to fight overfitting. The basic idea behind regularization is to constraint the hypothesis sets to get better performance in  $E_{out}$

### 1. Intuitive Explanation

Remember the VC bound we have:


$$
E_{o u t}(g) \leq E_{i n}(g)+O(\sqrt{d_{V C} \frac{\log (N / \delta)}{N}})
$$


When we increase the complexity of model, we could get lower and lower $E_{in}(g)$. However, the increase of $d_{vc}$ (Notice that this time $d_{vc}$ refer to the effect number of parameter, not the VC dimension for hypothesis sets.)  would eventually increase the $E_{out}(g)$.  But there is the chance that by sacrificing a little bit $E_{in}(g)$, we can dramatically let $d_{vc}$ decrease, which would cause  $E_{out}(g)$ decrease. This is the crux of regularization. By constraining the learning algorithm to select "simpler" hypothesis from $\mathcal{H}$, we sacrifice a little bias for a significant gain in the variance.



### 2. Soft-order Constraints

First we introduct soft-order constraints:


$$
H(c)=\{h\in H_{Q}\ and\ \sum^{Q}_{q=0} w_{q}^2\le c\}
$$


Where $H_{Q}$ is Q-order polynomial hypothesis(which can be easily generalized to other learning algorithm).  There are two properties we can see: first, when $c\rightarrow \infty$, $H(c)=H_{Q}$. Second, when $c_1\le c_2$, we have $d_{vc}(H(c_1))\le d_{vc}(H(c_2))$. 

Two main questions for this is how to choose $c$ and given $c$, how to find $g$. The first question related to validation. We now try solve the second question. We can further formalize the constraints to a optimization problem:


$$
\min _{w} E_{\mathrm{in}}(w) \quad \text { subject to } \quad w^{\mathrm{T}} w \leq c
$$


we denote the result as $w_{reg}$ and the optimization problem without constraints as $w_{lin}$.

When $c$ is bigger enough, that is $w_{lin}^Tw_{lin}\le c$, we have $w_{lin} =w_{reg}$.

when $c$ is not bigger enough, that is  $w_{lin}^Tw_{lin}\ge c$,  we actually have

1. $w_{reg}^Tw_{reg}=c$.  Since if $w_{reg}^Tw_{reg} \le c$, we can improve $E_{in}(w)$ by increase $w_{reg}^Tw_{reg}$, which would finally casue $w_{reg}^Tw_{reg}=c$. 

2. $w_{r e g} \propto-\nabla_{w} E_{i n}\left(w_{r e g}\right)$, which means, $w_{reg}$ have the same direction with $-\nabla_{w} E_{i n}\left(w_{r e g}\right)$. To see this, we can utilize follow plot:

   <img src="/assets/images/image-20200223115158168.png" alt="image-20200223115158168" style="zoom:67%;" />

If $w_{r e g}$ don't have same direction with $ -\nabla_{w} E_{i n}\left(w_{r e g}\right)$, which show in the plot. In the next step of optimization, we can divide the gradient to two direction, one is parallel with $w_{reg}$, one is tangent with error circle where $w_{reg}^Tw_{reg}=c$. This indicates we can further minimize $E_{in}$ with holding $w_{reg}^Tw_{reg}=c$.  Only if $w_{r e g} \propto-\nabla_{w} E_{i n}\left(w_{r e g}\right)$, we cannot find tangent direction, which means it is true.

So, we can find some constant $\lambda_c$ such that


$$
\nabla_{w} E_{i n}\left(w_{r e g}\right)=-\frac{2 \lambda_{C}}{N} w_{r e g}
$$


$$
\nabla_{w}\left(E_{i n}\left(w_{r e g}\right)+\frac{\lambda_{C}}{N} w_{r e g}^{T} w_{r e g}\right)=0
$$



Which means that, $w_{reg}$ is the solution for $E_{i n}(w)+\frac{\lambda_{C}}{N} w^{T} w$. 

Now, we successfully transform the original constrainted optimization problem to a unconstrainted problem:


$$
\text { minimize } E_{i n}(w)+\frac{\lambda_{C}}{N} w^{T}w
$$


Notes that use Lagrange multiplier, we can get same formula.When $w$ is linear model, we call this **Ridge Regression​**

We define **augmented error** as:


$$
E_{a u g}(w)=E_{i n}(w)+\frac{\lambda_{c}}{N} w^{T} w
$$
$

w^Tw$ also called weight decay regularizer. To see why,  we can written the gradient descent update rule for this:


$$
\begin{array}{l}
{w(t+1) \leftarrow w(t)-\eta \nabla_{w} E_{a u g}(w(t))} \\
{\Rightarrow w(t+1) \leftarrow\left(1-2 \eta \lambda_{C}\right) w(t)-\eta \nabla_{w} E_{i n}(w(t))}
\end{array}
$$


The term $\left(1-2 \eta \lambda_{C}\right)$ can be seen as the decay of the weight.

### 3. LASSO

Normally, in soft-order constraints linear model, or Ridge regression, we can penalty for a large $w$. However, seem like it never force any $w$ to zero, which make the  interpretation of coefficient hard. Now, we introduce another regularizer which can solve this problem:


$$
\Omega=\sum^{Q}_{q=0}|w_{q}|
$$


The unconstrained form of optimization problem is:


$$
\text { minimize } E_{i n}(w)+\frac{\lambda_{C}}{N}\sum^{Q}_{q=0}|w_{q}|
$$


This is called **LASSO regression**.

<img src="/assets/images/image-20200223122418448.png" alt="image-20200223122418448" style="zoom:50%;" />

The comparation between LASSO and Ridge regression is illustrated in above plot. We can see LASSO is more likely end up with four corner of rhombus, that's why LASSO can make some coefficients to become zero.

### 4. General Form of Regularization

We define general form of regularization as 


$$
E_{a u g}(h, \lambda, \Omega)=E_{i n}(\vec{w})+\frac{\lambda}{N} \Omega(h)
$$


where $\Omega$ is the regularizer, $\lambda$ is amount of regularization. Notice that this formula look really like the VC bound:


$$
E_{o u t}(g) \leq E_{i n}(g)+O(\sqrt{d_{v c} \frac{\ln N}{N}})
$$


Which means, if we carefully choose $\Omega$, we can better approach $E_{out}(g)$ by $E_{aug}(g,\lambda,\Omega)$.



### 5. More on Regularization

In this part, we briefly discuss the impact of regularization in different situations.

First, let's see the impact for different $\lambda$:

<img src="/assets/images/image-20200223155646528.png" alt="image-20200223155646528" style="zoom:50%;" /> 

When $\lambda$ is small, the constraints we put is small. So model would focus on the $E_{in}$ term, which would lead to overfitting. With the increasing of $\lambda$, the variance introduced by large parameters decrease lead to a better $E_{out}$. However, when $\lambda$ is too large, we put so much constraints on it, which introduce much bais and cause $E_{in}$ increase. This is underfitting.

In the right plot, we use a little bit different regularizer called low order, which tend to make high order parameter be smaller. Since this further force high order to be smaller, a small amout of $\lambda$ could cause model to be underfitting.

Next, let's see the impact for different types of target function(There, both target function and model have same order.):

<img src="/assets/images/image-20200223160947183.png" alt="image-20200223160947183" style="zoom:50%;" />

The left plot show the target function with different stochastic noise. When $\sigma^2$  is 0, we can see regularization don't help, cause there don't have noise. when $\sigma^2$ goes larger, the amount of regularization we need also become large. Right plot show the impact of different order for target function. We can see the same pattern as stochastic noise.





### Reference

[1] Learning From Data, Abu-Mostafa, Magdon-Ismail, and Lin.

[2] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

