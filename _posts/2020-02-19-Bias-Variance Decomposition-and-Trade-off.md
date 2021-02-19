---
title: "Bias-Variance Decomposition and Trade-off"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide

---



### 1. Decomposition

Now, instead consider classification problem which error is binary error, we consider real-value function, let's say regression, we define mean square error as follow:


$$
e(h(\vec{x}), f(\vec{x}))=\left(h\left(\vec{x}_{n}\right)-f\left(\vec{x}_{n}\right)\right)^{2}
$$


Then, the in-sample and out-sample error is:


$$
E_{i n}(g)=\frac{1}{N} \sum_{n=1}^{N} e\left(h\left(\vec{x}_{n}\right), f\left(\vec{x}_{n}\right)\right)=\frac{1}{N} \sum_{n=1}^{N}\left(h\left(\vec{x}_{n}\right)-f\left(\vec{x}_{n}\right)\right)^{2}
$$


$$
E_{\text {out}}(g)=\mathbb{E}_{\vec{x}}\left[e\left(h\left(\vec{x}_{n}\right), f\left(\vec{x}_{n}\right)\right)\right]=\mathbb{E}_{\vec{x}}\left[(g(\vec{x})-f(\vec{x}))^{2}\right]
$$



 Now, what can we say about $E_{out}(g)$?

Since $g$ is derived by a specific data set $D$, thus we use $g^{(D)}$ here,


$$
E_{o u t}\left(g^{(D)}\right)=\mathbb{E}_{\bar{x}}\left[\left(g^{(D)}(\vec{x})-f(\vec{x})\right)^{2}\right]
$$


let $\bar{g}(\vec{x})=\mathbb{E}_{D}[g^{(D)}(\vec{x})]$, then we can compute $\mathbb{E}_D [E_{out}]$:


$$
\begin{array}{l}
{\mathbb{E}_{D}\left[E_{o u t}\left(g^{(D)}\right)\right]} \\
{=\mathbb{E}_{D}\left[\mathbb{E}_{\vec{x}}\left[\left(g^{(D)}(\vec{x})-f(\vec{x})\right)^{2}\right]\right]} \\
{=\mathbb{E}_{\vec{x}}\left[\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})+\bar{g}(\vec{x})-f(\vec{x})\right)^{2}\right]\right]}\\
{=\mathbb{E}_{\vec{x}}\left[\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)^{2}+(\bar{g}(\vec{x})-f(\vec{x}))^{2}+2\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)(\bar{g}(\vec{x})-f(\vec{x}))\right]\right]}
\end{array}
$$


Note that$\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)\right]=0$, thus we have : 


$$
\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)(\bar{g}(\vec{x})-f(\vec{x}))\right]=(\bar{g}(\vec{x})-f(\vec{x})) \mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)\right]=0
$$


Therefore:


$$
\begin{array}{l}
{\mathbb{E}_{D}\left[E_{o u t}\left(g^{(D)}\right)\right]} \\
{=\mathbb{E}_{\vec{x}}\left[\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)^{2}+(\bar{g}(\vec{x})-f(\vec{x}))^{2}\right]\right]} \\
{=\mathbb{E}_{\vec{x}}\left[\mathbb{E}_{D}\left[\left(g^{(D)}(\vec{x})-\bar{g}(\vec{x})\right)^{2}\right]\right]+\mathbb{E}_{\vec{x}}\left[(\bar{g}(\vec{x})-f(\vec{x}))^{2}\right]} \\
{=\mathbb{E}_{\vec{x}}\left[\text { Variance of } g^{(D)}(\vec{x})+\operatorname{Bias} \text { of } \bar{g}(\vec{x})\right]} \\
{=\text { Variance }+\text { Bias }}
\end{array}
$$


This is decomposition of variance and bias, However, since we don't know $\bar{g}$ and $f$, we cannot calculate variance and bias, this just a conceptual decomposition.



#### 2. Bias-Variance Trade-off

In prctice, we define Variance and bias as follow:

**Bias**: Bias refers to the error that is introduced by modeling a complicated real-world problem by a much simpler model.

**Variance**: Variance refers to how much your estimate for $f$ would change by if you had a different training data set.

Since our goal is to minimize the out-sample error $E_{out}(g)$, that is minimize both variance and bias. However, turns out  it is impossible to minimize two at same time. Let's discuss what variance and bias would go by change follow.

**1. Number of data**

when we fix the $H$, and increase the number of data, we would expect variance goes down and bias roughly stay the same, and we expect $E_{out}$ decrease

**2. Complexity of Hypothesis Sets**

When we increase the complexity of $H$, we would expect the variance increase and bias decrease.  This is called variance-bais trade-off.

**3. Learning Curve**

<img src="/assets/images/image-20200212004042680.png" alt="image-20200212004042680" style="zoom:25%;" />

Above picture describe the relationship between the comlexity of $H$, number of data points with error.



<img src="/assets/images/image-20200212004111254.png" alt="image-20200212004111254" style="zoom:25%;" />



### Reference

[1] Learning From Data, Abu-Mostafa, Magdon-Ismail, and Lin.

[2] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

