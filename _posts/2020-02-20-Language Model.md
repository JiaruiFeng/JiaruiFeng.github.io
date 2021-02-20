---
title: "Language Model"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide

---



### 1.Concept

The Language Model is use to assign a probability to a sequence of tokens. We can start with an example:

<center>"The cat jumped over the puddle."</center>

This is definitely a valid sentence. A good language model should give this sentence a high probability. On contary, the language model should give invalid sentence a low probability. Bascially, there are two common type of language model, the first is Statistical language model, the second is Nerual Network Language Model(NNLM)

### 2.Statistical Language Model

Suppose we have a sentence with length $m$, Mathematically, the probability of this sentence is:


$$
P(w_1,w_2,...,w_m)
$$


By the chains rule, we can calculate the value of above:


$$
P(w_1,w_2,...,w_m)=P(w_1)P(w_2|w_1)P(w_3|w_2,w_1)···P(w_i|w_{i-1},w_{i-2},..,w_1)
···P(w_m|w_{m-1}...w_1)
$$


However, in practial, this is hard to calculate. Now, we can introduce N-gram model to simplify the calculation.

### 3.N-gram Model

The main idea of N-gram model is that: Suppose the probability of word $i$ only depends on the previous $n-1$ words, thus the conditional probability $P(w_i\lvert w_{i-1},w_{i-2},..,w_1)$ can be simplify as following:


$$
P(w_i|w_{i-1},w_{i-2},..,w_1)=P(w_i|w_{i-1},w_{i-2},..,w_{i-(n-1)})
$$


If we let $n=1$, it become **Unigram Model**:


$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=1}^{n} P\left(w_{i}\right)
$$


That is, the probability of each word in sentence only depends on itself. However, In practice, it always not the case.

If we let $n=2$, it become **Bigram Model​**:


$$
P\left(w_{1}, w_{2}, \cdots, w_{n}\right)=\prod_{i=2}^{n} P\left(w_{i} | w_{i-1}\right)
$$



In practice, the probability is computed by the equation:


$$
P(w_i|w_{i-1},w_{i-2},..,w_{i-(n-1)})=\frac{count(w_i,w_{i-1},w_{i-2},..,w_{i-(n-1)})}{count(w_{i-1},w_{i-2},..,w_{i-(n-1)})}
$$


There are sparsity problems with this equations. First, if the count of $(w_i,w_{i-1},w_{i-2},..,w_{i-(n-1)})$ is zero in corpus, the probability of this would be always zero.To solve this, a small $\delta$ could be added to the count for each word in the vocabulary. This is called *smoothing*. Second, Secondly, consider the denominator of Equation, if count of $(w_{i-1},w_{i-2},..,w_{i-(n-1)})$  is zero in corpus. for all of the $w_{i}$, there are no probability can be calculated. To solve this, we could consider reduce the number of $n$ in this case. This is called *backoff*. Increasing $n$ makes problems worse. Normally, we would like to set $n<5$.

### 4.Nerual Network Language Model(NNLM)

In 2003, some researchers suggest that use nerual network to construct language model. Different from traditional statistical language model, NNLM use nerual network to estimate the probability of  $n$ words' sentence directly. The architecture of nerual network is following:

![img](/assets/images/20160922200454004.png)

Bascially, the procedures of NNLM are: first, collect all the sentences  $ w_{1 }...
...w_{T},
w_{t} \in D$ with length $n$ from corpus, $D$ is the collection of sentences. The objective function of NNLM is to calcuate the probability of word $w_i$ when the input sentence is $w_{i-(n-1)},···,w_{i-1}$:


$$
\sum_{w_i\in D} P(w_i|w_{i-(n-1)},···,w_{i-1})
$$


The input of model is low-dimension vector(mention in part three) of words joint together. The output is a vector with length $V$ represent the probability of every word in corpus appear as $w_i$, $V$ is the number of words in corpus. 



### 5. Recurrent Neural Network Language Model(RNN-LM)

Even with NNLM, we still compute probability within specific size of window, which would definitely lose information. Now, we introduce recurrent neural network language model(RNN-LM), which is capable of conditioning the model on all previous words in corpus.

For the theory of RNN, you can see this post: [Recurrent Neural Network](https://jiaruifeng.github.io/DeepLearning/RNN.html). Here we only talk about how to build RNN-LM.

<img src="/assets/images/image-20200311000117367.png" alt="image-20200311000117367" style="zoom:50%;" />

Above is the architecture of RNN-LM. In each time step, we input a word embedding of current word, then model will compute the probability of the next word. Here is the detailed procedures:

* Let $x_{1}, \ldots, x_{t-1}, x_{t}, x_{t+1}, \ldots x_{T}$ be the word vectors corresponding to a corpus with $T$ words, where $x_{t} \in \mathbb{R}^{d}$.

* In each time step $t$, compute hidden feature $h_{t}=\sigma\left(W_{h} h_{t-1}+W_{e} x_{t}\right)$ , where $\sigma$ is the activation function, $h_{t-1} \in \mathbb{R}^{D_h}$ is output of the non-linear function at the previous time-step $t-1$, $W_{h} \in \mathbb{R}^{D_h\times D_h}$ is weights matrix used to condition the output of previous time-step $h_{t-1}$, $W_{e} \in \mathbb{R}^{D_h\times d}$ is weights matrix used to condition the input word vector $x_t$.
* After we have $h_t$, we can use $$\hat{y}_{t}=\operatorname{softmax}\left(Uh_{t}\right)$$ to calcuate the probability distribution over the vocabulary at each time-step $t$, where $$\hat{y}_t \in \mathbb{R}^V$$,$V$ is the number of words in corpus. 

In RNN-LM, we always use cross-entropy as loss function. The total loss for a sentence should be the summation of the loss for every time step, which is:


$$
J=\frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)=-\frac{1}{T} \sum_{t=1}^{T} \sum_{j=1}^{|V|} y_{t, j} \times \log \left(\hat{y}_{t, j}\right)
$$


We can aslo define **Perplexity** to evaluate the performance of language model, which is:


$$
Perplexity =2^{J}
$$


it is basically 2 to the power of the negative log probability of the cross entropy error function.

RNN-LM have several advantages:

1. RNN-LM can use in input sequence with any length.
2. The size of model wouldn't be influent by increasing the length of sequence.
3. It can "remember" words in all previous time steps.
4. The same weights are applied to every timestep of the input, so there is symmetry in how inputs are processed.

However, RNN-LM also have drawback like heavy computation and gradient vanishing.



### Reference

[1] CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html