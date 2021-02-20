---
title: "Attention Mechanism"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide
---



### 1. Problem with Seq2seq

A bottleneck for seq2seq is that we encoder all the information for an input sectence to a context vector. Which is actually hard in practical, especially when the setcence is too long.  Meanwhile, different words in sectence definitely have different importance and different parts of the output may even consider different parts of the input "important". A very smart way to solve above problems is **Attention Mechanism**.



### 2. Introduction of Attention

The main idea of attention is that: on each step of decoder, use direct connection to the encoder to focus on praticular part of input sectence.

**Encoder**

Let $(h_1, . . . , h_n)$ be the hidden vectors representing the input sentence, dimension is $d_1$. These vectors are the output of each time step and capture the contextual representation for each input word. 

**Decoder**

We want to compute the output $y_i$ in time step $i$ of the decoder using a recursive formula of the form:


$$
y_{i}=f\left(s_{i}, c_{i}\right)
$$


where $s_{i}$ is the hidden state in current time step,dimension is $d_2$, $y_{i}$ is the output words for previous hidden state, $c_i$ is the context vector that capture relevant information in time step $i$, which is unlike to original seq2seq. The procedure of getting $c_i$ is follow:

* Compute the attention score $e_i$ where:

  

$$
e_{i, j}=a\left(s_{i}, h_{j}\right)
$$



$a$ is a function. There are several ways to implement this function:

**Basic dot-product attention**: $e_{i,j}=s_{i}^Th_j$, This is the way in the original attention paper. Notice that, this way assume the dimesion of encoder hidden state and decoder hidden state is equal.

**Multiplicative attention**: $e_{i,j}=s_{i}^T Wh_j$, where $W \in \mathbb{R}^{d_2\times d_1}$. 

**Additive attention**: $e_{i,j}=v^{T} \tanh \left(W_{1} h_{j}+W_{2} s_{i}\right)$, where $W_{1}\in \mathbb{R^{d_3 \times d_1}}$, $W_2 \in \mathbb{R}^{d_3 \times d_2}$.



* Then, use softmax function to get normalize score:
  
  
  $$
\alpha_{i, j}=\frac{\exp \left(e_{i, j}\right)}{\sum_{k=1}^{n} \exp \left(e_{i, k}\right)}
  $$
  
  
* Compute context vector for time step $i$:
  
  
  $$
c_{i}=\sum_{j=1}^{n} \alpha_{i, j} h_{j}
  $$
  
  
* Finally we concatenate the attention output with the decoder hidden state and proceed as in the non-attention seq2seq model:
  
  
  $$
  y_{i}=f\left(\left[s_{i}, c_{i}\right]\right)
  $$



Here is a picture show how attention work:

![image-20200317191707227](/assets/images/image-20200317191707227.png)

This also called **Global Attention Model**. Meanwhile, There also have **Local Attention Model**:

![image-20200317191840101](/assets/images/image-20200317191840101.png)

the model first predicts a single aligned position $p_t$ for the current target word. A window centered around the source position $p_t$ is then used to compute a context vector $c_t$.

### 3. Advantages for Attention

1. Attention significantly improve the performance of nerual machine translation problem, since it help the model to focus on specific part of input.

2. Attention solve the information bottleneck problem in original seq2seq.

3. Attention help to alleviate the gradient vanishing problem since it construct more directly connected layers.

4. Attention provide a way for people to interpret model use attention distribution:

   <img src="/assets/images/image-20200317191249847.png" alt="image-20200317191249847" style="zoom:50%;" />

5. Attention is not only work for machine translation, but many sequence problem.

**General Definition of Attention**:Given a set of vector values, and a vector query, **attention** is a technique to compute a weighted sum of the values, dependent on the query.



### Reference

[1] CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html

[2]Luong, M.-T.; Pham, H. & Manning, C. D. (2015), 'Effective approaches to attention-based neural machine translation', *arXiv preprint arXiv:1508.04025* .