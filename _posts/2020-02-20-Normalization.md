---
title: "Normalization"
categories:
  - Deep Learning
tags:
  - Learning note
classes: wide

---



### 1. Internal Covariate Shift

One of the challenges of deep learning is that the gradients with respect to the weights in one layer are highly dependent on the outputs of the neurons in the previous layer especially if these outputs change in a highly correlated way. In other word, the parameter change in previous layer will cause the distribution of the input in next layer change, this is called **internal covariate shift(ICS)**.  ICS will cause the next layer to adjust themselves according to the input from previous layer during training, which will slow down the speed of training. To deal with this, some normalization methods are proposed.

### 2. Batch Normalization

Batch normlization is an idea to prevent internal covariate shift and speed up neural network training. Batch normalization take the idea whitening the input of neural network as to normalize the input of each layer. However, the problem occur with whitening input is that, first, the computation cost to whitening input in each layer is expensive. Second, note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To deal with these issues, batch normalization decide to normalize data in mini-batch size and also add some trick to maintain the information expression ability of data. Here is the formula of batch normalization. Let value of input in layer $l$ over mini-batch be $$\mathcal{B}=\{x_{1 \ldots m}\}$$, $\gamma$ and $\beta$ are gain parameters that use to restore the ability of data to represent information:


$$
\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i}\\
\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}\\
\widehat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\\
y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right)
$$


Where $$\widehat{x}_{i}$$ is normalized $$x_i$$, $$\epsilon$$ is a constant to prevent variance to be zero. During each mini-batch of training, we can calculate $$\mu_\mathcal{B}$$ and $$\sigma_{\mathcal{B}}$$ , then use this term to normalize output. $$\gamma$$ and $$\beta$$ are learnable parameters.

##### Using batch normalization during prediction

However, during prediction or testing, we usually have fewer or even one data. This time we cannot calculate mean and variance based on such few dataset. What we do is that, After we train the model, we will save all the $\mu_{\mathcal{B}}$ and $\sigma_{\mathcal{B}}^{2}$ during training for each batch noramlized layer. The mean and variance for testing is:


$$
\mu_{t e s t}=\mathbb{E}\left(\mu_{\mathcal{B}}\right)\\
\sigma_{\text {test}}^{2}=\frac{m}{m-1} \mathbb{E}\left(\sigma_{\mathcal{B}}^{2}\right)
$$


After we get the unbiased estimation of mean and variance, we can do batch normalization using follow formula:


$$
B N\left(X_{t e s t}\right)=\gamma \cdot \frac{X_{t e s t}-\mu_{t e s t}}{\sqrt{\sigma_{t e s t}^{2}+\epsilon}}+\beta
$$



### 3. Layer Normalization

A big drawback of batch normalization is that the performance of batch normalization highly depend on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. Recently, a new neural network normalization method proposed to deal with such problem called layer normalization

Notice that changes in the output of one layer will tend to cause highly correlated changes in the summed inputs to the next layer, especially with ReLU units whose outputs can change by a lot. This suggests the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer. 

Assume for node $i$ in layer $l$, the weight is $w_i^l$, The input of this layer is $h^l$. Then, the output for the node is:


$$
a^l_i={w^l}^T_ih^l\\
h^{l+1}_i=f(a^l_i+b^l_i)
$$


Where $f$ is activation function and $b^l_i$ is the constant bias. Next, let $H$ be the number of node in layer $l$, Layer normalization is follow:


$$
\mu^{l}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{l}\\
\sigma^{l}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{l}-\mu^{l}\right)^{2}}\\
\bar{a}_{i}^{l}=\frac{g_{i}^{l}}{\sigma_{i}^{l}}\left(a_{i}^{l}-\mu_{i}^{l}\right)+b
$$


Where $\bar{a_i}^l$ is normalized output, $g^l_i$ is a gain parameter scaling the normalized activation before the non-linear activation function, $b$ is bias. In layer normalization, all the hidden units in a layer share the same normalization terms $\mu$ and $\sigma$, but different training cases have different normalization terms.

##### Use Layer Normalization in RNN

The problem for batch normalization apply in RNN is that, For RNN, we alway have different number of time step in different data. For, batch normalization, we need to store the $\mu$ and $\sigma$ in each time step. However, during testing, we may face that the number of time step larger than any training example. Instead, layer normalization don't have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step. In standard RNN, the formula for one time-step is:


$$
a^t=W_{hh}h^{t-1}+W_{xh}x^t
$$


The formula of layer normalization for RNN is:
$$
\mu^{t}=\frac{1}{H} \sum_{i=1}^{H} a_{i}^{t}\\
\sigma^{t}=\sqrt{\frac{1}{H} \sum_{i=1}^{H}\left(a_{i}^{t}-\mu^{t}\right)^{2}}\\
\mathbf{h}^{t}=f\left[\frac{\mathbf{g}}{\sigma^{t}} \odot\left(\mathbf{a}^{t}-\mu^{t}\right)+\mathbf{b}\right]
$$


Where $\mathbf{h}^{t}$ is normalized output, $\mathbf{g}$ and $\mathbf{b}$ is gain parameter and bias. $\odot$ is element-wise multiplication between two vectors. Layer normalization can also used to solve gradient vanishing and exploding problem for RNN. In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics.



### Reference

[1] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 2015.

[2] Ba, Jimmy, Jamie Ryan Kiros and Geoffrey E. Hinton. Layer Normalization. *ArXiv*abs/1607.06450 (2016): n. pag.