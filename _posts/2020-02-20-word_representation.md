---
title: "Word Representation"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide


---



## 1 Simple Word Representation
### 1.1 One-Hot Vector
One-hot vector is probably the most simple way to represent words. Basically,the one-hot vector represent every word as an $\mathbb{R}^{|V| \times 1}$  vector with all 0s and 1 at the index of the word we want to represent. $|V|$ is the number of words in our vocabulary. such a vector would look like following:


$$
w^{a n d v a r k}=\left[\begin{array}{c}
{1} \\
{0} \\
{0} \\
{\vdots} \\
{0}
\end{array}\right], w^{a}=\left[\begin{array}{c}
{0} \\
{1} \\
{0} \\
{\vdots} \\
{0}
\end{array}\right], w^{a t}=\left[\begin{array}{c}
{0} \\
{0} \\
{1} \\
{\vdots} \\
{0}
\end{array}\right], \ldots w^{z e b r a}=\left[\begin{array}{c}
{0} \\
{0} \\
{0} \\
{\vdots} \\
{1}
\end{array}\right]
$$


There are several problems with this representation. First, every word vector is totally independent entity, which we cannot calculate the similarity between two vector. For instance,


$$
\left(w^{h o t e l}\right)^{T} w^{\text {motel}}=\left(w^{h o t e l}\right)^{T} w^{c a t}=0
$$


Second, vectors could become very large when our vocabulary become large. When our dictionary have $10^6$  words, the matrix of words vector is $10^6 \times 10^6$ , which is cost to store and do further computing. 

### 1.2 Term Frequency-Inverse Document Frequency(TF-IDF)

TF-IDF is composed of two part, Term Frequency(TF) and Inverse Document Frequency(IDF).

#### Term Frequency(TF)

literally,  TF is the frequency a word $i$ appear in a specific document $j$. The Formula is following:


$$
\mathrm{tf}_{\mathrm{i}, \mathrm{j}}=\frac{n_{i, j}}{\sum_{k} n_{k, j}}
$$


$n_{i, j}$ Is the time of word $i$ appear in document $j$, $k$ is the number of words in the document.

#### Inverse Document Frequency(IDF)

IDF is the frequency a word $i$ appear in every document in corpus. The formula is following:


$$
\mathrm{idf}_{\mathrm{i}}=\lg \frac{|D|}{\left|\left\{j: t_{i} \in d_{j}\right\}\right|}
$$


$|D|$ is the total number of documents in corpus, $d_{j}$ is the document $j$. Combine TF and IDF, we can now define the TF-IDF as following:


$$
\operatorname{tfid} \mathrm{f}_{\mathrm{i}, \mathrm{j}}=\mathrm{tf}_{\mathrm{i}, \mathrm{j}} \times \mathrm{idf}_{\mathrm{i}}
$$


TF-IDF can capture the high frequent words in each document. It is always used to find key words in document. Meanwhile, by multipling IDF, TF-IDF avoid capture some common preposition or article. However, TF-IDF failed to capture context of every word in document but only consider the frequency of words.

### 1.3 Window Based Co-occurence Matrix

The window based co-occurence matrix can help to capture information in the context of words. $X$ stores co-occurrences of words thereby becoming an affinity matrix. In this method we count the number of times each word appears inside a window of a particular size around the word of interest. We calculate this count for all the words in corpus. We display an example below. Let our corpus contain just three sentences and the window size be 1:

1. I enjoy flying.
2. I like NLP.
3. I like deep learning.

The resulting counts matrix will then be:

![image-20200119230233520](/assets/images/image-20200119230233520.png)



## 2 Word Embedding
### 2.1 Concept

a word embedding is a vector representation of words, in some space $\mathbb{R}^k$.  That it, for every word in the encoder, the word embedding represents this word with a $k$-dimensional vector.  There are a few important differences, though:

1. Euclidean distances in the word embedding (attempt to) correspond to some notion of similarity between the words.
2. The dimensionality of the vector, $k$, does not need to be the same as the vocabulary size (typically it is a much smaller fixed dimension, like $k = 300$.

Word embedding is usually seem like following:

![image-20200120000845561](/assets/images/image-20200120000845561.png)

As seen in the figure, "similar" words are mapped to points close together (by Euclidean distance), whereas dissimilar words will be further away.

### 2.2 Word2Vec

Word2Vec a set of powerful algorithms to help us get the word embedding. 

#### 2.2.1 Basical Idea

Suppose we have a large corpus, every word in a fixed vocabuary is represented by a vector. Go through each position $t$ in text, which has a center word $c$ and context words $o$. Then, we can use the similarity of vector $c$ and $o$ to calculate the probability  of $o$ given $c$. Finally, by optimizing the parameters in model to maximize above probability, we can get optimal word embedding. The likelihood function is following:


$$
L(\theta)=\prod_{t=1}^{T} \prod_{-m \leq j \leq m \atop j \neq 0} P\left(w_{t+j} | w_{t} ; \theta\right)
$$


$T$ is the total number of position in text. $m$ is context window. $\theta$ are parameters in model. By log and average the likelihood function, our objective function is:


$$
J(\theta)=-\frac{1}{T} \log L(\theta)=-\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m} \log P\left(w_{t+j} | w_{t} ; \theta\right)
$$


The final problem is how to calculate $\log P\left(w_{t+j} \lvert w_{t} ; \theta\right)$ . Here, we define two vectors for every word $w$, $v_w$ when word $w$ is center word, and $u_w$ when word w is context word. Then, for the center word $c$ and context word $o$, the probability $P(o\lvert c)$ is:


$$
P(o | c)=\frac{\exp \left(u_{o}^{T} v_{c}\right)}{\sum_{w \in V} \exp \left(u_{w}^{T} v_{c}\right)}
$$


Use optimization techniques like gradient descent, we can optimize $\theta$ (Here, $\theta$ are all the vectors $v$ and $u$.). Then we can get our word embedding vector $u$ and $v$.

There are two classical model of Word2Vec, the one is Continuous Bags of Word(CBOW) and the other is Skip-gram model.

#### 2.2.2 Continuous Bags of Word(CBOW) Model

The basical idea behind CBOW model is, given the surronding context words, predict the probability of center word.

The input of CBOW model are one hot vectors or context we will represent with an $x^{(c)}$. the ouput is also one-hot vector of center word, define as $y$.  Then, we can define the arichtecture CBOW model as following:



<img src="/assets/images/image-20200120011426053.png" style="zoom:70"/>

The CBOW follow the basical idea of Word2Vec, so here we define to matrix $\mathcal{V} \in \mathbb{R}^{n \times\lvert V\rvert}$ and $
\mathcal{U} \in \mathbb{R}^{\lvert V\rvert  \times n}
$ , n is the size of word embedding vector, which we can define by ourselves. $\mathcal{V}$  is the input word matrix such that the $i$-th column of $\mathcal{V}$ isthe $n$-dimensional embedded vector for word $w_i$ when it is input of model. we denote this $n \times 1$ vector as $v_i$.   Similarly, $\mathcal{U}$ is the output matrix, the $j$-th row of $\mathcal{U}$ is the output vector of word $w_j$, we denote this row of $\mathcal{U}$ as $u_j$. We can find that, in the figure above, the $W_{V\times N}$ is actually the input matrix $\mathcal{V}$ and $W_{N \times V}^{\prime} $ is the output matrix $\mathcal{U}$. 

Now, the procedures of CBOW can be divided into follow part:

1. We generate our one hot word vectors for the input context of size $m:\left(x^{(c-m)}, \ldots, x^{(c-1)}, x^{(c+1)}, \ldots, x^{(c+m)} \in \mathbb{R}^{|V|}\right)$ .
2. Then, we compute the emdedding vector for our context by $\left(v_{c-m}=\right.\left.\mathcal{V} x^{(c-m)}, v_{c-m+1}=\mathcal{V} x^{(c-m+1)}, \ldots, v_{c+m}=\mathcal{V} x^{(c+m)} \in \mathbb{R}^{n}\right)$.
3. Average these vectors to get $\hat{\mathcal{v}}=\frac{v_{c-m}+v_{c-m+1}+\ldots+v_{c+m}}{2 m} \in \mathbb{R}^{n}$.
4. generate a score vector $z=\mathcal{U} \hat{v} \in \mathbb{R}^{|V|}$. As the dot product of similar vectors is higher, it will push similar words close to each other in order to achieve a high score.
5. Turn the scores into probabilities $\hat{y}=\operatorname{softmax}(z) \in \mathbb{R}^{|V|}$.
6. We desire our probabilities generated,$\hat{y} \in \mathbb{R}^{\lvert V\rvert}$ , to match the true probabilities,$y \in \mathbb{R}^{\lvert V\rvert }$, which also happens to be the one hot vector of the actual word.

Next step, we need to define our objective function. As we mentioned above, we want the probability of center word in $\hat{y}$ to be maximized. Thus, we can use cross-entropy to measure to different distribution:


$$
H(\hat{y}, y)=-\sum_{j=1}^{|V|} y_{j} \log \left(\hat{y}_{j}\right)
$$


Since the $y$ is actually a one-hot vector with only center word $y_i$ equal to 1 and others equal to 0, we can simplify our formula:


$$
H(\hat{y}, y)=- \log \left(\hat{y}_{i}\right)
$$


We thus formulate our optimization objective as:


$$
\begin{aligned}
\text { minimize } J &=-\log{\hat{y_i}} \\
&=-\log P\left(w_{c} | w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m}\right) \\
&=-\log P\left(u_{c} | \hat{v}\right) \\
&=-\log \frac{\exp \left(u_{c}^{T} \hat{v}\right)}{\sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)} \\
&=-u_{c}^{T} \hat{v}+\log \sum_{j=1}^{|V|} \exp \left(u_{j}^{T} \hat{v}\right)
\end{aligned}
$$


Use optimization techniques, we can update all relevant word vectors $u_c$ and $v_j$.

After trainning, we can get the result of word embedding by $\mathcal{V}$ and $\mathcal{U}$. we can find these are just the weights of hidden layer and weights of output layer in nerual network. For example, suppose $n=300$, the matrix $\mathcal{V}$  looks like:

![image-20200120105302771](/assets/images/image-20200120105302771.png)

#### 2.2.3 Skip-gram Model

Skip-gram model the another implementation of Word2Vec. What it does is just the opposite of CBOW. given the center word $c$, we want tp predict the context words $o$ of $c$.

The input one hot vector (center word) we will represent with an $x$ (since there is only one). And the output vectors as $y^{(j)}$. We define $\mathcal{V}$ and $\mathcal{U}$ the same as CBOW. The architecture of Skip-gram model is following:

<img src="/assets/images/image-20200120105705863.png" style="zoom:60"/>

We breakdown the way this model works in these 6 steps:

1. We generate our one hot input vector $x \in \mathbb{R}^{\lvert V\rvert}$ of the center word.
2. We get our embedded word vector for the center word $v_{c}=\mathcal{V} x \in \mathbb{R}^{n}$.
3. Generate a score vector $z=\mathcal{U} v_{c}$.
4. Turn the score vector into probabilities, $\hat{y}=\operatorname{softmax}(z)$. Note that $\hat{y}_{c-m}, \ldots, \hat{y}_{c-1}, \hat{y}_{c+1}, \ldots, \hat{y}_{c+m}$ are the probabilities of observeing each context word.
5. We desire our probability vector generated to match the true prob- abilities which is $y^{(c-m)}, \ldots, y^{(c-1)}, y^{(c+1)}, \ldots, y^{(c+m)}$, the one hot vectors of the actual output.

Here, in order to define objective function, we need to introduce an assumption, which is also the main difference between CBOW and Skip-gram. we assume that given the center word, all output words are completely independent, this is called Naive Bayes assumption. Thus, our objective function can be defined as following:


$$
\begin{aligned}
\text { minimize } J &=-\log P\left(w_{c-m}, \ldots, w_{c-1}, w_{c+1}, \ldots, w_{c+m} | w_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(w_{c-m+j} | w_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} P\left(u_{c-m+j} | v_{c}\right) \\
&=-\log \prod_{j=0, j \neq m}^{2 m} \frac{\exp \left(u_{c-m+j}^{T} v_{c}\right)}{\sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right)} \\
&=-\sum_{j=0, j \neq m}^{2 m} u_{c-m+j}^{T} v_{c}+2 m \log \sum_{k=1}^{|V|} \exp \left(u_{k}^{T} v_{c}\right)
\end{aligned}
$$


The basical CBOW and Skip-gram have a big disadvantage. Running gradient descent on a neural network that large is time consuming. And to make matters worse, we need a huge amount of trainning data in order to tune that many weights and avoid over-fitting. Researchers improved these by two way: Negative sampling and Hierarchical softmax. 

### 2.3 Hierarchical Softmax Based Word2Vec

#### 2.3.1 Concept 

The majoirty of computation would happen in output layer and softmax function when corpus goes to large and large. Hierarchical softamx offer an alleviant to it. 

First, researchers improve the structure of model. In the input to hidden layer, we don't use linear transformation and activation function. Instead, we only do summation of all the input word vectors:

![image-20200120114003188](/assets/images/image-20200120114003188.png)

The second improvement is to use Huffman tree to replace softmax layer. So, how do we project our word vector to Huffman tree?

Basically,Each leaf of the tree is a word, and there is a unique path from root to leaf. In this model, there is *no output representation for words*. Instead, each node of the graph (except the root and the leaves) is associated to a vector that the model is going to learn.

<img src="/assets/images/image-20200120114500291.png" style="zoom:50"/>

In this model, the probability of a word $w$ given a vector $x_{w}$,$P\left(w \lvert  x_{w}\right)$ , is equal to the probability of a random walk starting in the root and ending in the leaf node corresponding to *w*. The main advantage in computing the probability this way is that the cost is only $O(log(\lvert V\rvert ))$, corresponding to the length of the path. another advantage is that, since Huffman tree always let the high frequent word to be closed to the root, thus we would expect to spend less time in high frequent words.

So, how to calculate $P\left(w \lvert  w_{i}\right)$? in Hierarchical softmax, we use sigmoid function. We define that if it walk to the left leaf, it is negative(Huffman coding is 1), if it walk to the right leaf, it is positive(Huffman coding is 0). the positive probability can be defined as following:


$$
P(+)=\sigma\left(x_{w}^{T} \theta\right)=\frac{1}{1+e^{-x_{w}^{T} \theta}}
$$


$\sigma(\cdot)$ Is sigmoid function, $x_w$ is the word embedding, $\theta$ is parameters of sigmoid function in the current node. use the property of sigmoid function, the negative probability is:


$$
P(-)=1-P(+)
$$


for the $w_2$ in the above, we want to maximize the $P(-)$ of $n(w_2,1)$ and $n(w_2,2)$, $P(+)$ of $n(w_2,3)$:


$$
max P\left(w_2 | x_{w_2}\right)=\prod_{i=1}^{3} P\left(n\left(w_{2}\right), i\right)=\left(1-\frac{1}{1+e^{-x_{w_2}^{T} \theta_{1}}}\right)\left(1-\frac{1}{1+e^{-x_{w_2}^{T} \theta_{2}}}\right) \frac{1}{1+e^{-x_{w_@}^{T} \theta_{3}}}
$$


Note that, use sigmoid function, we can also ensure that $\sum_{w=1}^{\lvert V\rvert } P\left(w \lvert  x_{w}\right)=1$, just like original softmax.

Now, we define the length of node $w_i$ is the length from root to the leaf node which contain $w_i$, denote as $l_w$. from the root, the Huffman coding of every node on the path of $w$ is denoted by $d_{i}^{w} \in\{0,1\}, i=1,2,...,l_w-1$. Then, we can define the probability for $w$ to path through node $j$ as:


$$
P\left(d_{j}^{w} | x_{w}, \theta_{j-1}^{w}\right)=\left\{\begin{array}{ll}
{\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} & {d_{j}^{w}=0} \\
{1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} & {d_{j}^{w}=1}
\end{array}\right.
$$


Thus, for word $w$, the likelihood is:


$$
\prod_{j=2}^{l_{w}} P\left(d_{j}^{w} | x_{w}, \theta_{j-1}^{w}\right)=\prod_{j=2}^{l_{w}}\left[\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]^{1-d_{j}^{w}}\left[1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]^{d_{j}^{w}}
$$


$$
L=\log \prod_{j=2}^{l_{w}} P\left(d_{j}^{w} | x_{w}, \theta_{j-1}^{w}\right)=\sum_{j=2}^{l_{w}}\left(\left(1-d_{j}^{w}\right) \log \left[\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]+d_{j}^{w} \log \left[1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]\right)
$$



use gradient ascending, we can get the gradient of $\theta_{j-1}^{w}$ :


$$
\begin{aligned}
\frac{\partial L}{\partial \theta_{j-1}^{w}} &=\left(1-d_{j}^{w}\right) \frac{\left(\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right)\right.}{\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} x_{w}-d_{j}^{w} \frac{\left(\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right.\right.}{1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} x_{w} \\
&=\left(1-d_{j}^{w}\right)\left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) x_{w}-d_{j}^{w} \sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right) x_{w} \\
&=\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) x_{w}
\end{aligned}
$$


the same as $\theta_{j-1}^{w}$, the gradient of $x_w$ is:


$$
\frac{\partial L}{\partial x_{w}}=\sum_{j=2}^{l_{w}}\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) \theta_{j-1}^{w}
$$



#### 2.3.2 Hierarchical Softmax Based CBOW 

Suppose our word embedding is of size $M$, the context size is $2c$. In order to use hierarchical softmax based CBOW, here are the steps:

1. Build a Huffman tree for our corpus $M$

2. From the input layer to projection layer, simply do summation and average of the vectors of $2c$ context words:
   
   
   $$
x_{w}=\frac{1}{2 c} \sum_{i=1}^{2 c} x_{i}
   $$
   
   
3. Use gradient ascend to update $\theta_{j-1}^{w}$ and $x_w$. Note here $x_w$ is the summation and average of all $2c$ words, after updating, we can use the gradient to update original $x_i(i=1,2,...,2c)$ :

   
   $$
   \begin{array}{c}
   {\theta_{j-1}^{w}=\theta_{j-1}^{w}+\eta\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) x_{w}} \\
   {x_{i}=x_{i}+\eta \sum_{j=2}^{l_{w}}\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) \theta_{j-1}^{w}(i=1,2 \ldots, 2 c)}
   \end{array}
   $$
   

   $\eta$ is the learning rate.

    Thus, let $e=0$ , for j=2 to $l_w$, we can calculate:

   ​		

$$
f=\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)
$$

$$
g=\left(1-d_{j}^{w}-f\right) \eta
$$

$$
e=e+g \theta_{j-1}^{w}
$$

$$
\theta_{j-1}^{w}=\theta_{j-1}^{w}+g x_{w}
$$

​	

​	Finally, Update word vectors for every context word $x_i$:


$$
x_{i}=x_{i}+e
$$



#### 2.3.3 Hierarchical softmax based Skip-gram

In the Skip-gram model, input only have one word $x_w$ and output are $2c$ context words. we hope the value of hierarchical softmax for these $2c$ words are bigger than other. Here are the steps:

1. Build a Huffman tree for our corpus $M$

2. From the input layer to projection layer, we only have one word, thus $x_w$ is the output of projection layer.

3. Use gradient ascend to update $\theta_{j-1}^{w}$ and $x_w$. Note here we actually have $2c$ vector in the context. we hope to maximize $P(x_i\lvert x_w), i=1,2,...,2c$. Meanwhile, we also want to maximize $P(x_w\lvert x_i),i=1,2,...,2c$. the Word2Vec use the latter one. cause the iteration for every vector would become more balanced.

   for $i=1$ to $2c$:

   ​	e=0 

   ​	for j=2 to $l_w$, we calculate:
   
   
   $$
f=\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)
   $$
   
   $$
g=\left(1-d_{j}^{w}-f\right) \eta
   $$
   
   $$
e=e+g \theta_{j-1}^{w}
   $$
   
   $$
\theta_{j-1}^{w}=\theta_{j-1}^{w}+g x_{w}
   $$
   
   ​		
   
   ​	Finally, Update word vectors for every context word $x_i$:
   $$
   x_{i}=x_{i}+e
   $$



### 2.4 Negative Sampling Based Word2Vec

Hierarchical softmax reduce the computation of original Word2Vec. However, it is not good when we want to calculate a word with low frequency, since it would show up in very deep leaf. Negative Sampling use another way to solve this problem.

For every training step, instead of looping over the entire vocabulary, we can just sample several negative examples.  Then, we can update our parameters based on the one positive sample and several negative sample. Suppose we have $Neg$ negative sample $(context(w),w_i),i=1,2,...,Neg$, the postitve sample is $w_0$. we hope to maximize the probability of postivie sample and minimize the probability of negative sample, that is maximize $1-P$:


$$
max P\left(\text { context }\left(w_{0}\right), w_{i}\right)=\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right), y_{i}=1, i=0
$$




$$
max P\left(\text { context }\left(w_{0}\right), w_{i}\right)=1-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right), y_{i}=0, i=1,2, \dots \text { neg }
$$



We can write it down in following form:


$$
max \prod_{i=0}^{n e g} P\left(\text {context}\left(w_{0}\right), w_{i}\right)=\sigma\left(x_{w_{0}}^{T} \theta^{w_{0}}\right) \prod_{i=1}^{n e g}\left(1-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right)
$$


From the logistic regression, we can write down our likelihood function:


$$
\prod_{i=0}^{n e g} \sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)^{y_{i}}\left(1-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right)^{1-y_{i}}
$$


$$
L=\sum_{i=0}^{n e g} y_{i} \log \left(\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right)+\left(1-y_{i}\right) \log \left(1-\sigma\left(x_{w_{0}}^{T} \theta_{i}^{w_{i}}\right)\right)
$$

The same as Hierarchical softmax, we use gradient ascending to update  $\theta^{w_{i}}$ and $x_w$. The gradient of $\theta^{w_{i}}$ is:


$$
\begin{aligned}
\frac{\partial L}{\partial \theta^{w_{i}}} &=y_{i}\left(1-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right) x_{w_{0}}-\left(1-y_{i}\right) \sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right) x_{w_{0}} \\
&=\left(y_{i}-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right) x_{w_{0}}
\end{aligned}
$$


The gradient of $x^{w_0}$ is:


$$
\frac{\partial L}{\partial x^{w_{0}}}=\sum_{i=0}^{n e g}\left(y_{i}-\sigma\left(x_{w_{0}}^{T} \theta^{w_{i}}\right)\right) \theta^{w_{i}}
$$


How do we determine our negative samples? basically, they are selected using a "Unigram distribution", where most frequent words are more likely to be selected as negative samples:


$$
P\left(w_{i}\right)=\frac{f\left(w_{i}\right)^{3 / 4}}{\sum_{j=0}^{n}\left(f\left(w_{j}\right)^{3 / 4}\right)}
$$


$P(w_i)$ is the probability of word $w_i$ to be selected as negative sample. $f(w_i)$ is the frequency of word $w_i$.

The way this selection is implemented in the C code is interesting. They have a large array with 100M elements (which they refer to as the unigram table). They fill this table with the index of each word in the vocabulary multiple times, and the number of times a word’s index appears in the table is given by $P(wi) \times table size$. Then, to actually select a negative sample, you just generate a random integer between 0 and 100M, and use the word at that index in the table. Since the higher probability words occur more times in the table, you’re more likely to pick those. Here is the illustration:

![image-20200120142312415](/assets/images/image-20200120142312415.png)

The procedures of Negative sampling based CBOW and Skip-gram are the same as Hierarchical softmax based CBOW and Skip-gram,  just substitue all the leaf nodes by negative and positive sample.

 

### Reference

[1]http://web.stanford.edu/class/cs224n/index.html

[2]http://www.datasciencecourse.org

[3]Mikolov, Tomas , et al. "Efficient Estimation of Word Representations in Vector Space." *Computer Science* (2013).

[4]Mikolov, Tomas , et al. "Distributed Representations of Words and Phrases and their Compositionality." *Advances in Neural Information Processing Systems* 26(2013):3111-3119.

[5] http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

[6]https://blog.csdn.net/WFRainn/article/details/83718244

[7]https://www.cnblogs.com/pinard/p/7243513.html

