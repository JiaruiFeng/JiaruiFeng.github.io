---
title: "Backpropagation of CNN"
categories:
  - Deep Learning
tags:
  - Learning note
classes: wide



---



### 1. Notation

Assume layer is $$l=1,2,...,L$$, data in each layer is $$X^{(l)}$$, where $$x^{(l)}_{ij}$$ is the value in the position $(i,j)$ of $$X^{(l)}$$, $$i,j=1,2,...,d^{(l)}$$, $d^{(l)}$ is dimension of data in layer $l$. Let size of filter be $$s*s$$ and $$k_{pq}^{(l)}$$ is the weight of filter in position $(p,q)$ in layer $l$, where $$p,q=1,2,...,s$$. $f$ is activation function and $b^{(l)}$ is the constant in layer $l$. Finally, the loss function is $L$. Notice that all the derivation is base on stride equal to 1.

### 2. Convolutional layer

For convolutional layer,   we can first write forward propagation formula:


$$
X^{(l)}_{ij}=f(u^{(l)}_{ij})=f(\sum^s_{p=1}\sum^s_{q=1}x^{(l-1)}_{i+p-1,j+p-1}k_{pq}^{(l)}+b^{(l)})
$$


In back propagation, we want to calculate gradient of $$k^{(l)}_{pq}$$. Notice that each position of output is associate with every $$k^{(l)}_{pq}$$ and $$b^{(l)}$$. To calculate gradient, we write:


$$
\begin{aligned}
\frac{\partial L}{\part k_{pq}^{(l)}}&=\frac{\part L}{\part X^{(l)}}\frac{\part X^{(l)}}{\part u^{(l)}}\frac{\part u^{(l)}}{\part k_{pq}^{(l)}}\\
&=\sum^{d^{(l)}}_{i=1}\sum^{d^{(l)}}_{j=1}\frac{\part L}{\part x^{(l)}_{ij}}\frac{\part x^{(l)}_{ij}}{\part u^{(l)}_{ij}}\frac{\part u^{(l)}_{ij}}{\part k_{pq}^{(l)}}
\end{aligned}
$$


$$
\begin{aligned}
\frac{\part L}{\part b^{(l)}}&=\frac{\part L}{\part X^{(l)}}\frac{\part X^{(l)}}{\part u^{(l)}}\frac{\part u^{(l)}}{\part b^{(l)}}\\
&=\sum^{d^{(l)}}_{i=1}\sum^{d^{(l)}}_{j=1}\frac{\part L}{\part x^{(l)}_{ij}}\frac{\part x^{(l)}_{ij}}{\part u^{(l)}_{ij}}\frac{\part u^{(l)}_{ij}}{\part b^{(l)}}
\end{aligned}
$$



From the forward formula, we can see that:


$$
\frac{\part x^{(l)}_{ij}}{\part u^{(l)}_{ij}}=f^{\prime}(u^{(l)}_{ij})\\
\frac{\part u^{(l)}_{ij}}{\part k_{pq}^{(l)}}=x^{(l-1)}_{i+p-1,j+p-1}\\
\frac{\part u^{(l)}_{ij}}{\part b^{(l)}}=1
$$


Let set $$\delta_{ij}^{(l)}=\frac{\part L}{\part u^{(l)}_{ij}}=\frac{\part L}{\part x^{(l)}_{ij}}\frac{\part x^{(l)}_{ij}}{\part u^{(l)}_{ij}}$$, we have:


$$
\frac{\part L}{\part k_{pq}^{(l)}}=\sum^{d^{(l)}}_{i=1}\sum^{d^{(l)}}_{j=1}\delta_{ij}^{(l)} x^{(l-1)}_{i+p-1,j+p-1}\\
\frac{\part L}{\part b^{(l)}}=\sum^{d^{(l)}}_{i=1}\sum^{d^{(l)}}_{j=1}\delta_{ij}^{(l)}
$$


Actually, if we close look at formula of $\frac{\part L}{\part k_{pq}^{(l)}}$, we notice that it is doing the convolution operation, we can wirte it as:


$$
\nabla_{k^{(l)}}L=conv(X^{(l-1)},\delta^{(l)})
$$


Next, we want to calculate $\delta^{(l)}$ in a recursive formula. We can write:


$$
\delta_{ij}^{(l)}=\frac{\part L}{\part x^{(l)}_{ij}}\frac{\part x^{(l)}_{ij}}{\part u^{(l)}_{ij}}\\
\delta_{ij}^{(l-1)}=\frac{\part L}{\part x^{(l-1)}_{ij}}\frac{\part x^{(l-1)}_{ij}}{\part u^{(l-1)}_{ij}}\\
$$
​	

Notice that $$\delta^{(l-1)}_{ij}$$ is only associate with part of $$x^{(l)}_{ij}$$. Here we give some examples. Suppose we want calculate $$\delta^{(l-1)}_{11}$$:


$$
\delta^{(l-1)}_{11}=\frac{\part L}{\part x^{(l-1)}_{ij}}\frac{\part x^{(l-1)}_{ij}}{\part u^{(l-1)}_{ij}}=(\sum^{d^{(l)}}_{i=1}\sum^{d^{(l)}}_{j=1}\delta^{(l)}_{ij}\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{11}})\frac{\part x^{(l-1)}_{11}}{\part u^{(l-1)}_{11}}
$$


Since $x_{11}^{(l-1)}$ only be used to calculate $u^{(l)}_{11}$,


$$
\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{11}}=k_{11}^{(l)}\ \ \ \ \ \ \ i,j=1\\
\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{11}}=0\ \ \ \ \ \ \ otherwise
$$


Similarly, we can calculate $x^{(l-1)}_{12}$:


$$
\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{12}}=k_{12}^{(l)}\ \ \ \ \ \ \ i,j=1\\
\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{12}}=k_{11}^{(l)}\ \ \ \ \ \ \ i=1,j=2\\
\frac{\part u^{(l)}_{ij}}{\part x^{(l-1)}_{12}}=0\ \ \ \ \ \ \ otherwise
$$


Finally, we find rule of recursive formula:


$$
\delta^{(l-1)}=\delta^{(l)}_p\ rot180(k^{(l)})\odot f^{\prime}(u^{(l-1)})
$$


where $\delta^{(l)}_p$ is $\delta^{(l)}$ with 2-padding surrounding, $\odot$ is point-wise production. $rot180$ is rotate matrix for 180 degree. For example:


$$
\left[\begin{array}{lll}k_{11} & k_{12} & k_{13} \\ k_{21} & k_{22} & k_{23} \\ k_{31} & k_{32} & k_{33}\end{array}\right]\rightarrow\left[\begin{array}{lll}k_{33} & k_{32} & k_{31} \\ k_{23} & k_{22} & k_{21} \\ k_{13} & k_{12} & k_{11}\end{array}\right]
$$



### 3.  Pooling Layer

In pooling layer, we find that we don't have parameters that need to train and update, instead, we only need to propagate gradient to next layer. First, we wirte the formula of forward propagation:


$$
X^{(l)}=down(X^{(l-1)})
$$


Thus, in back propagation, we actually want upsampling:


$$
\delta^{(l-1)}=up(\delta^{(l)})
$$


Suppose the pooling size is $s$, we first calculate average pooling:


$$
y^{(l)}=\frac{1}{s\times s}\sum^{k}_{i=1}x_i^{(l-1)}
$$


$y$ is the output of average pooling. In back propagation:


$$
\frac{\part L}{\part x_i^{(l-1)}}=\frac{\part L}{\part y^{(l)}}\frac{\part y^{(l)}}{\part x_i^{(l-1)}}=\frac{1}{s\times s}\delta^{(l)}
$$


So, we just expend gradient to each position to form a block, where the size of block is $s*s$:


$$
\left[\begin{array}{ccc}
\frac{\delta^{(l)}}{s \times s} & \dots & \frac{\delta^{(l)}}{s \times s} \\
\dots & \dots & \dots \\
\frac{\delta^{(l)}}{s \times s} & \dots & \frac{\delta^{(l)}}{s \times s}
\end{array}\right]
$$


If we use max pooling:


$$
y^{(l)}=max(x_1^{(l-1)},x_2^{(l-1)},...,x_l^{(l-1)})=x_i^{(l-1)}
$$

$$
\frac{\part L}{\part x_{i}^{(l-1)}}=\frac{\part L}{\part y^{(l)}}\frac{\part y^{(l)}}{\part x^{(l-1)}_i}=\delta^{(l)}\frac{\part y^{(l)}}{\part x^{(l-1)}_i}
$$



if $x^{(l-1)}_i$ is max, $\frac{\part y^{(l)}}{\part x^{(l-1)}_i}=1$, else =0.

### 4. Fully-connected Layer

The back propagation of fully-connected layer is actually the same as what we derive in neural network. Thus we have all the things to do back propagation in CNN.



### Reference

[1] tensorinfinity. http://www.tensorinfinity.com



