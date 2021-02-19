---
title: "MLE,MAP and Bayesian Estimation"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide


---



Basically, In the field of machine learning, given the dataset $D=\{X,Y\}$ and parameter $\theta$, What we want to do is to estimate $\theta$ based on $D$.

 There are three type of estimation that can help us to find the above probaility. Here, we assume that every $x^{(i)} \in X$ is generated independently and $y^{(i)}$ only dependent on $x^{(i)}$ and noise.

### 1. Maximum Likelihood Estimation(MLE)

The maximum likelihood estimator is defined by conditional density $f_{Y|X}(y|x)$, the Formula is follow:


$$
g_{MLE}(y):=\underset{x}{\operatorname{argmax}} f_{Y | X}(y | x)
$$


Now, for the estimation of $\theta$:


$$
\begin{aligned}
p(D;\theta)&=p(x^{(1)},x^{(2)},...,x^{(n)})p(y^{(1)},y^{(2)},...,y^{(n)}|x^{(1)},x^{(2)},...,x^{(n)};\theta)\\
&=\prod^{N}_{i=1}p(x^{(i)})\prod^{N}_{i=1}p(y^{(i)}|x^{(i)};\theta)
\end{aligned}
$$


Since the probability of $P(x^{(i)})$ is constant for specific dataset, the maximum likelihood estimation of $\theta$ is :


$$
\theta_{MLE}:=\underset{\theta}{\operatorname{argmax}}\prod^{N}_{i=1}p(y^{(i)}|x^{(i)};\theta)
$$


In this sense, the $\theta$ is constant value but unknownis taken in **frequentist statistics**. Our job is try to estimate this parameter.

### 2. Bayesian Estimation

In bayesian view, we can think $\theta$ as being a random variable whose value is unknown. In this approach, we would specify a prior distribution $P(\theta) $ on $\theta$ that express our "prior beliefs" about the parameters. Given the  dataset $D$,


$$
\begin{aligned}
p(\theta | D) &=\frac{p(D | \theta) p(\theta)}{p(D)} \\
&=\frac{\left(\prod_{i=1}^{n} p\left(y^{(i)} | x^{(i)}, \theta\right)\right) p(\theta)}{\int_{\theta}\left(\prod_{i=1}^{n} p\left(y^{(i)} | x^{(i)}, \theta\right) p(\theta)\right) d \theta}
\end{aligned}
$$


Now, give a new test example $x$, we can compute our posterior distribution on the $y$ using the posterior distribution on $\theta$:


$$
\begin{aligned}
p(y|x,D)&=\int_{\theta} p(y,\theta|x,D)d\theta\\
&=\int_{\theta}p(y|x,\theta,D)p(\theta|x,D)d\theta\\
\end{aligned}
$$


Since $\theta$ is determined by $D$ and not rely on $x$:


$$
\begin{aligned}
p(y|x,D)&=\int_{\theta}p(y|x,\theta)p(\theta|D)d\theta
\end{aligned}
$$


$$
E[y|x,D]=\int_{y}yp(y|x,D)dy
$$



This is bayesian estimation. However, in reality, in order to compute the $E[y|x,D]$, we need consider all possible $\theta$, which is time consuming. 



### 3. Maximum Posterior Probability Estimation(MAP)

Notice that in bayesian esitmation, $p(D)$ would not infulence estimation of $p(\theta|D)$. Thus, the estimation of $\theta$ can be written as:


$$
\begin{aligned}
\theta_{MAP}&=\underset{\theta}{\operatorname{argmax}}p(\theta|D)\\
&=\underset{\theta}{\operatorname{argmax}}\prod^{N}_{i=1}p(y^{(i)}|x^{(i)};\theta)p(\theta)
\end{aligned}
$$


We can see this is the same formula as for the MLE estimation for $\theta$ except the prior probability of $\theta$. If $p(\theta)$ is uniform distribuion, we have$\theta_{MLE}=\theta_{MAP}$.

In practical applications, a common choice for $p(\theta)$ is $\theta \sim N(0,\sigma^2 I)$. In practice, this causes the Bayesian MAP estimate to be less susceptible to overfitting than the ML estimate of the parameters.



### Reference

[1] CS229: Machine Learning, Stanford. http://cs229.stanford.edu 

[2] Probability and Random Processes for Electrical and Computer Engineers,by John A. Gubner, Cambridge University Press, 2006.  