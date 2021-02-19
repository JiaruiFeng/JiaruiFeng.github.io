---
title: "Neural Network"
categories:
  - Deep Learning
tags:
  - Learning note
classes: wide

---



Neural networks are a biologically inspired model which has had considerable engineering success in applications ranging from time series prediction to vision.

### 1. Structure of Neural Network

##### Perceptrons

First, we discuss structure of single perceptron. Define $x^{(i)}\in \mathbb{R}^{d\times 1}$ is the input data, $b$ is a constant, $f$is activation function, which we will talk about more later, the structure of perceptrons is

<img src="/assets/images/image-20200418184048396.png" alt="image-20200418184048396" style="zoom:50%;" />

Where $w_1,w_2,...,w_d$ are the weight which we need to learn. The $u$ is calculated by follow:


$$
u=w_1x^{(i)}_1+w_2x^{(i)}_2+...+w_dx^{(i)}_d+b
$$


Which is exactly the form of linear function.

##### Neural Network

Next, if we combine multiple perceptrons, we can get the structure of neural network

<img src="/assets/images/image-20200418185258263.png" alt="image-20200418185258263" style="zoom:50%;" />

Typically, a neural network would have three type of layer. the first is input layer, we input the data to input layer. Then follow layers except last layer are hidden layers. the final layer is output layer, here output what we need. They connected by different weighted egdes. The dimension of input layer usually is the dimension of data. the dimension and number of hidden layer are defined by us. the dimension of output depend on the problem. for regression problem, demension is one. for the multiple classification problem, the dimension of output usually equal to the number of categories.

##### Activation Function

What is activation function and why we need it? Actually, if we take deep look at the neural network, we can see that, if we don't apply activation function, even if we have multiple hidden layers, we are still doing linear function. How do we convert nerual network to a non-linear form, that's where activation function comes in. Usually, activation function is a non-linear function, here are some common activation functions:

![image-20200418191005751](/assets/images/image-20200418191005751.png)

### 2. Notation and Matrix Form of Nerual Network

In order to better illustrate the calculation of nerual network, we first introduce the notation of neural network. Since nerual network usually implemented in matrix form in programming language, we would like to written down it in matrix form.

Let us define the input data as $X\in \mathbb{R}^{d^{(0)\times N}}$, where $d^{(0)}$ is the dimension of data and $N$ is the number of data. Then we define each layer in network as $l=0,1,2,...,L$, where $l=0$ means input layer, $l=L$ means output layer. Dimension of each layer is $d^{(l)}$. Thus, the input of layer $l$ is $X^{(l-1)} \in \mathbb{R}^{d_{(l-1)}\times N}$ . The weights of each layer is $W^{(l)}\in \mathbb{R}^{d^{(l)}\times d^{(l-1)}}$. Finally, we define the output of nerual network as $h(X)$.  With all the notation, we can rewrite the function for each layer in nerual network:


$$
u^{(l)}=W^{(l)}X^{(l-1)}\\
X^{(l)}=f(u^{(l)})
$$



### 3. Forward Propagation

**Forward propagation** is how nerual network get outpu with input data. we can compute $X^{(l)}$ layer by layer and finally get output $h(X)$:


$$
X=X^{(0)}\rightarrow u^{(1)}\rightarrow X^{(1)} \rightarrow...\rightarrow u^{(L)}\rightarrow X^{(L)}=h(X)
$$



### 4. Backward Propagation

The real challange is how to update and learn the weight based on forward propagation. In many other algorithm, we use gradient descent. However, it's not the case in nerual network. Since we have many weight in different layer, we cannot write down a single form of weight. Thus we need to find a better way. That's what backward propagation's job.

We first use MSE loss function and regression problem setting to illstruate algorithm, since it have a uniform matrix form. 

Definr the loss function as:


$$
E=\frac{1}{2N}||Y-h(X)||^2
$$


Where $Y\in \mathbb {R}^{N\times 1}$ is the true value. take the derivate of loss function:


$$
\frac{\part E}{\part h(X)}=\frac{1}{N}(h(X)-Y)
$$



##### Output Layer

We define that $\frac{\part E}{\part u^{(l)}}=\delta^{(l)}$, for output layer, we have


$$
\delta^{(L)}=\frac{\part E}{\part u^{(L)}}=\frac{\part E}{\part h(X)}\frac{\part h(X)}{\part u^{(L)}}
=\frac{1}{N}(h(X)-Y)\odot f^{\prime}(u^{(L)})
$$


Where $\odot $ is element-wise multiplication. For the gradient of $W^{(L)}$ and $b^{(L)}$, we can get:


$$
\nabla_{W^{(L)}}=\frac{\part E}{\part u^{(L)}}\frac{\part u^{(L)}}{\part W^{(L)}}=\delta^{(L)}{X^{(L-1)}}^T\\
\nabla_{b^{(L)}}=\frac{\part E}{\part u^{(L)}}\frac{\part u^{(L)}}{\part b^{(L)}}=\delta^{(L)}
$$



##### Hidden Layer

For hidden layer, we can first write forward propagation as:


$$
u^{(l+1)}=W^{(l+1)}X^{(l)}=W^{(l+1)}f(u^{(l)})
$$


Thus:


$$
\begin{aligned}
\delta^{(l)}=\frac{\part E}{\part u^{(l)}}&=\frac{\part E}{\part u^{(l+1)}}\frac{\part u^{(l+1)}}{\part f(u^{(l)})}\frac{\part f(u^{(l)})}{u^{(l)}}\\
&={W^{(l+1)}}^T\delta^{(l+1)} \odot f^{\prime}(u^{(l)})
\end{aligned}
$$


For the gradient of $W^{(l)}$ and $b^{(l)}$, we have the same form:


$$
\nabla_{W^{(l)}}=\delta^{(l)}{X^{(l-1)}}^{T}\\
\nabla_{b^{(l)}}=\delta^{(l)}
$$



##### Procedure of Algorithm

Until now, we have the boundary case of $\delta^{(l)}$ and recursive form, we can calculate gradient of every weight layer by layer and update weights use gradient descent, that is why we call it backward propagation.



##### Classification Problem Setting

Next, we briefly discuss how to derive the backward propagation for classification problem. Typically,  the only different between classfication and regression problem in neural network is lay down the output layer. Usually, we use softmax activation function in output layer and cross entropy function as loss function. Thus we use this setting to derive the back propagation formula. First, the formula of cross entropy is:


$$
E=\sum^P_{i=1}y_ilog\ h(x)_i
$$


Where $P$ is the number of output nodes,$P=d^{(L)}$, which is the number of categories we have. $x$ And $y$ are single data point and its true label(encode as one-hot vector). Take the derivate:


$$
\frac{\part E}{\part h(x)_i}=y_i\frac{1}{h(x)_i}
$$


Next, we define the input of output layer is $x^{(L)} \in \mathbb{R}^{d^{(L)}\times 1}$. we want to calculate gradient for $x^{L}$, we need first look at softmax function:


$$
f(x_k)=\frac{e^{x_k}}{\sum^d_{k=1}e^{x_k}}
$$


We can see the derivate of $x_j$ actually depend on all $x_j$ for $j=1,2,...,d$, use the chain rule:


$$
\frac{\part E}{\part u^{(L)}_j}=\sum^{P}_{i=1}\frac{\part E}{\part h(x)_i}\frac{\part h(x)_i}{f(u^{(L)}_j)}
$$


if $i==j$, we have:


$$
\frac{\part h(x)_i}{f(u^{(L)}_j)}=\left(\frac{e^{u^{(L)}_{i}}}{\sum_{k} e^{u^{(L)}_{k}}}\right)\left(1-\frac{e^{u^{(L)}_{i}}}{\sum_{k} e^{u^{(L)}_{k}}}\right)=h(x)_i(1-h(x)_i)
$$


if $i\ne j$, we have:


$$
\frac{\part h(x)_i}{f(u^{(L)}_j)}=-h(x)_ih(x)_j
$$


Thus:


$$
\begin{aligned}
\frac{\part E}{\part u^{(L)}_j}&=y_i\frac{1}{h(x)_i}h(x)_i(1-h(x)_i)-\sum^P_{i\ne j}y_i\frac{1}{h(x)_i}h(x)_ih(x)_j\\
&=y_i-y_ih(x)_i-\sum^P_{i\ne j}y_ih(x)_j
\end{aligned}
$$


Write down as matrix:


$$
\delta^{(L)}=\frac{\part E}{\part u^{(L)}_j}=
\left[
\begin{matrix}
y_1-y_1h(x)_1-\sum^P_{1\ne j}y_1h(x)_j\\
y_2-y_2h(x)_2-\sum^P_{2\ne j}y_2h(x)_j\\
...\\
y_P-y_Ph(x)_P-\sum^P_{P\ne j}y_Ph(x)_j
\end{matrix}
\right]
$$


With this boundary case, we can calculate gradient the same as what we derive above in regression setting, since except output layer, all the layers can use same format.



### 5. Limit of Neural Network

##### Gradient Vanishing 

If we take look at the form of back propagation, we can find in each step, we will time the $\delta^{(l)}$ with $f^{\prime}$, which is the derivate of activation function. However, this term is always less than 1, make the $\delta ^{(l)}$ smaller and smaller and hard to update if we have many layers. That is one of the biggest problem nerual network have before the emerge of deep learning.

##### Local Minimum

Since gradient descent can only guarantee to find local minimum if loss function is not convex function, we cannot promise it is the optimal solution. One way to deal with that is run neural network several times with different initial weights.



### Reference

[1] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

[2]tensorinfinity. http://www.tensorinfinity.com