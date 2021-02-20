---
title: "Optimization"
categories:
  - Deep Learning
tags:
  - Learning note
classes: wide

---



The optimization method in deep learning field has many difference with normal machine learning algorithm. First is that, many meachine learning algorithm have spcific loss function which is designed to be convex function. This make optimization become easy. However, in deep learning, loss function is always not convex, we cannot use convex optimization techique to solve it. Meanwhile, non-convex bring many local minimum into optimization, which make gradient-based algorithm much more diffcult to find a pleased solution. Here, we briefly talk about several optimization algorithm used in deep learning. Let $L(w,x,y)$ be the loss function, where $w$ are all the parameter in deep learning model, $x$ is input and $y$ is output.

### Gradient Descent

This is the most basic gradient-based optimization algorithm. the idea behind gredient-based algorithm is that, the opposite direction of gradient of a function is where a function can decrease most quickly.  The algorithm is follow:


$$
w:=w-\eta \nabla_w \frac{1}{N}\sum^{N}_{i=1}L(w,x_i,y_i)
$$


Where $N$ is the number of sample in training dataset, $\eta$ is learning rate, which is cruical in updating the weight. Notice that loss the is define by average the loss in total training dataset, which is actually a cost method especially when $N$ is very large. 

### Stochastic Gradient Descent(SGD)

To address the issue in gradient descent. stochastic gradient descent was proposed. Instead optmization weights amona all data, in each update, we only select one data sample and compute loss of this, then update weights based this loss:


$$
w:=w-\eta \nabla_w L(w,x_i,y_i)
$$


Stochastic gradient descent can start making progress right away, and continues to make progress with each example it looks at. 

Often, stochastic gradient descent gets $w$ “close” to the minimum much faster than gradient descent. Conside that the standard error of the mean estimated from $N$ samples is given by $\sigma/N$. where $\sigma$ is the true standard deviation of the value of the samples. Compare two hypothetical estimates of the gradient, one based on 100 examples and another based on 10,000 examples. The latter requires 100 times more computation than the former but reduces the standard error of the mean only by a factor of 10.

Meanwhile, estimate gradient from a small number of samples also reduce the redundancy in the training set. Conside that all $N$ samples in training set is identical, gradient can actually be computed by only one operation, with less $N-1$ times computation than naive approach.

### Minibatch Stochastic Gradient Descent

In practical, we always use more than one, but few than all the training sample, which called minibatch strochastic gradient descent. Let $M$ be the number of batch size:


$$
w:=w-\eta \nabla_w \frac{1}{M}\sum^{M}_{i=1}L(w,x_i,y_i)
$$


The reason to use minibatch instead of one sample is: First, training with small batch size might require a small learning rate to maintain stability because of the high variance in the estimate of the gradient. The total runtime can be very high as a result of the need to make more steps, both because of the reduced learning rate and because it takes more steps to observe the entire training set. Second, some kinds of hardware achieve better runtime with speciﬁc sizes of arrays. Especially when using GPUs, it is common for power of 2 batch sizes to oﬀer better runtime. Third, larger batches provide a more accurate estimate of the gradient. 

However, in parctical, the traditional gradient based algorithm encounter many problems. First is that it can hard deal with local minimum and saddle points. Second, if loss changes quickly in one direction and slowly in another, optimization will be very slow progress along shallow dimension, jitter along steep direction:

<img src="/assets/images/image-20200622235705466.png" alt="image-20200622235705466" style="zoom:50%;" />

Finally, learning rate dominate the performance of gradient descent in practical, a fixed learning rate is definitely not suitable for deep learning optimization, we need learning rate scheduling method. 

Thus, many improvements have be come up with to deal with such issues.

### SGD+Momentum

This method introduce "velocity" to decrease the variance in the optimization:


$$
v:= \nabla_w +\rho v\\
w:=w-\alpha v
$$


The momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction:

<img src="/assets/images/image-20200623000050508.png" alt="image-20200623000050508" style="zoom:50%;" />

 Previously, the size of the step was simply the norm of the gradient multiplied by the learning rate. Now, the size of the step depends on how large and how aligned a **sequence** of gradients are. The step size is largest when many successive gradients point in exactly the same direction. 

### Nesterov Momentum

Nesterov Momentum is a variant of the momentum algorithm that was inspired by Nesterov’s accelerated gradient method. Nesterov Momentum Look ahead to the point where updating using velocity would take us. Then compute gradient there and mix it with velocity to get actual update direction:

<img src="/assets/images/image-20200623001631997.png" alt="image-20200623001631997" style="zoom:50%;" />

 The update formula is given by:


$$
v_{t+1}=\rho v_{t}-\nabla_(w+\rho v_{t})\\
w_{t+1}=w_{t}+v_t
$$


Since $w+\rho v $ is not easy to compute, we cen let $\hat{w}=w+\rho v$, then:


$$
v_{t+1}=\rho v_t-\nabla_{\hat{w}_t}\\
\hat{w}_{t+1}=\hat{w}_t-\rho v_t +(1+\rho)v_{t+1}
$$



### AdaGrad

AdaGrad allow the learning rate to be adjusted based on the history of gradient. Meanwhile, each element in the weights would have their own learning rate:


$$
(g_{t+1})_i=(g_t)_i+(\nabla_{w_t})_i^2\\
(w_{t+1})_i=(w_{t})_i-\eta\frac{(\nabla_{w_t})_i}{\sqrt{(g_{t+1})}+\epsilon}
$$


$\epsilon$ is commonly be set to 1e-07. If history sum of gradients are large, then the learning rate will be small. However, the disadvantage of AdaGrad is that when $t$ goes up, the learning rate is tend to 0.

### RMSProp

AdaGrad is designed to converge rapidly when applied to a convex function. When applied to a nonconvex function to train a neural network, the learning trajectory may pass through many diﬀerent structures and eventually arrive at a region that is a locally convex bowl. RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after ﬁnding a convex bowl, as if it were an instance of the AdaGrad algorithm initialized within that bowl.


$$
(g_{t+1})_i=\epsilon (g_t)_i+(1-\epsilon)(\nabla_{w_t})_i^2\\
(w_{t+1})_i=(w_{t})_i-\eta\frac{(\nabla_{w_t})_i}{\sqrt{(g_{t+1})}+\epsilon}
$$



## Adam

Adam combine momentum and RMSProp together directly. Meanwhile, Adam includes bias corrections to the estimates of both the ﬁrst-order moments (the momentum term) and the (uncentered) second-order moments to account for their initialization at the origin(The initial value for first and second moments will be set to 0) :


$$
(m_{t+1})_i=\beta_1(m_{t})_i +(1-\beta_1)(\nabla_{w_t})_i\\
(v_{t+1})_i=\beta_2(v_{t})_i +(1-\beta_2)(\nabla_{w_t})_i^2\\
(\hat{m}_{t+1})_i=\frac{(m_{t+1})_i}{1-\beta_1^t}\\
(\hat{v}_{t+1})_i=\frac{(v_{t+1})_i}{1-\beta_2^t}\\
(w_{t+1})_i=(w_{t})_i-\eta\frac{(\hat{m}_{t+1})_i}{\sqrt{(\hat{v}_{t+1})_i}+\epsilon}
$$


Adam with$\beta_1 = 0.9$, $ \beta_2 = 0.999$, and $\eta = 1e-3\ or\ 5e-4$ is a great starting point for many models.

### Learning Rate Schedule

As we told, the learning rate is very important, to see that more carefully, here is one example:

<img src="/assets/images/image-20200623011715852.png" alt="image-20200623011715852" style="zoom:50%;" />

However, a appropriate learning rate is actually depend on dataset and nerual network architecture. Normally, we would like to start with a large learning rate, then decay it over time. Here are some common strategies used in learning rate schedule:

#### Step

Reduce learning rate at a few fixed points. For example, ResNets multiply learning rate by 0.1 after epochs 30, 60, and 90.

#### Cosine


$$
\eta_{t}=\frac{1}{2} \eta_{0}(1+\cos (\frac{t \pi} { T}))
$$



Where $\eta_0$ is initial learning rate, $\eta_t$ is learning rate at epoch $t$. $T$ is total number of epochs.

#### Linear


$$
\eta_t=\eta_0(1-\frac{t}{T})
$$



#### Inverse Sqrt


$$
\eta_t=\frac{\eta_0}{\sqrt{t}}
$$



### Reference

[1] deep learning book, http://www.deeplearningbook.org

[2] CS 231n Lecture notes,stanford. http://cs231n.stanford.edu

[3] tensorinfinity. http://www.tensorinfinity.com

 