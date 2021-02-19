---
title: "Overfitting"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide


---



Overfitting is that, **fit the data more than it warranted**, or fit the noise instead of trend.

The mean case is that, you pick up a hypothesis with very low $E_{in}$, however, the $E_{out}$ is higher. This mean that $E_{in}$ alone is no longer a good guide for learning. Consider a simple example:

<img src="/assets/images/image-20200220184101883.png" alt="image-20200220184101883" style="zoom:50%;" />

Suppose we have target function in 2-nd order with some noise, which is indicated in blue line.  There are 5 data points sampled from target function. Now, we use 4-th order function to fit the data, which is show in red line. Though the target function is simple, the 4-th order function use its power to fit the data very well, which result in a bad $E_{out}$. 

Overfitting do not only mean bad generalization:

<img src="/assets/images/image-20200220184848644.png" alt="image-20200220184848644" style="zoom:50%;" />

It is Going for lower and lower $E_{in}$ results in higher and higher $E_{out}$.

Consider another case:

<img src="/assets/images/image-20200220185840915.png" alt="image-20200220185840915" style="zoom:50%;" />

We have two target function, first one the 10-th order with some noise, second one is 50-th order with no noise. Now, we use 2-nd order hypothesis($H_2$) and 10-th order hypothesis($H_{10}$) to fit data respectively. Resutl is that:

<img src="/assets/images/image-20200220190031162.png" alt="image-20200220190031162" style="zoom:67%;" />

<img src="/assets/images/image-20200220190045947.png" alt="image-20200220190045947" style="zoom:67%;" />

We can see that in both cases, 10-th order function overfit the data. Instead, even though 2-nd order function do not capture the full nature of the target function, it do capture the trend.

<img src="/assets/images/image-20200220190338200.png" alt="image-20200220190338200" style="zoom:50%;" />

This picture show exactly what going on with two hypothesis sets. When data is not large, choosing $H_{10}$ would bring higher $E_{out}$ than $H_2$.

If we use $E_{\mathrm{out}}\left(H_{10}\right)-E_{\mathrm{out}}\left(H_{2}\right)$ to measure the overfit, we can get follow relationship :

![image-20200220191551360](/assets/images/image-20200220191551360.png)

We can see, 
$$
\begin{array}{ccc}
\hline \text { Number of data points } & {\uparrow} & {\text { Overfitting } \downarrow} \\
\hline \text { Noise } & {\uparrow} & {\text { Overfitting  }\uparrow} \\
\hline {\text { Target complexity }} & {\uparrow} & {\text { Overfitting }} \uparrow \\
\hline
\end{array}
$$
In the right chart, we can see a boundary when target complexity equal to 10. This because that, when the ture function form is close to the hypothesis sets form, we stand a better chance to learn it.

Stochastic noise refer to what we cannot decide and learn from the data, deterministic noise refer to the error we use a simple model to measure a much more complex question. Remeber the bias-variance decomposition, we have:
$$
\mathbb{E}_{\mathcal{D}}\left[E_{\text {out }}\right]=\sigma^{2}+\text { bias }+\text { var }
$$
The bias is actually deterministic noise and $\sigma^2$ is stochastic noise.



Typically, there are two way to fight overfitting: **regularization** and **validation**.



### Reference

[1] Learning From Data, Abu-Mostafa, Magdon-Ismail, and Lin.

[2] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

