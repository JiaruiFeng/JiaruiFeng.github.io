---
title: "LSA and GloVe"
categories:
  - Natural Language Processing
tags:
  - Learning note
classes: wide



---



### 1. Latent Semantic Analysis

Recall in the previous section we mention the co-occurence matrix used for word representation: Given a co-occurence matrix $X \in \mathbb{R}^{\lvert V\rvert \times \lvert V\rvert }$,where $X_{ij}$ indicates the number of times word $j$ occur in the context of word $i$. The window size of context is a hyper-parameter.

The idea behind **Latent Semantic Analysis(LSA)**  is simply that use low dimension word vector to represent the similarity between different words. Some introduction would use A matrix containing word counts per document or TF-IDF matrix to carry on LSA. But given the idea of LSA to mantain orignial similarity between words, it is also possible to use co-occurence matrix to do LSA. Thus, here we use co-occurence

With co-occurence matrix, what LSA does is just use SVD method to reduce dimensions. Given matrix $X$, apply SVD on $X$:


$$
X=U \Sigma V^{T}
$$


Then, By observing the most biggest $k$ singular values in $\sum$ , we can reduce the dimension of word vector from $\lvert V\rvert$ to $k$. Then the $U_{k} \in \mathbb{R}^{\lvert V\rvert \times k} $ would be our final word embedding.

Notice that since the number of column and the number of row is the same in co-occurence matrix, the $U$ and $V$ is actually the same. If we use matrix containing word counts per document, where $X \in \mathbb{R}^{\lvert V\rvert  \times m}$ with m represent the number of document, there would be some zero column in $\sum$, we need to discard them when doing dimension reduction:

 ![image-20200216183242128](/assets/images/image-20200216183242128.png)

Meanwhile, $U$ and $V$ are not the samething. $U$ is matrix represent every word, $V$ is matrix represent every document. Using $V$ and $V_{k}$, we can do some topic mining problem.



### 2. Global Vectors for Word Representation

#### Model

So far we have talking about count based word embedding(LSA) and direct prediction word embedding(Skip-gram and CBOW). However, both two of them have some disadvantages: for count based methods, it only capture the similarity between words, but it get bad prefermance in word analogy task. the direct prediction methods don't use the statistical information of words and are computation expensive. Here comes to the **Global Vectors for Word Representation(GloVe)**.

First, we define probability in co-occurence matrix $X$: Let $X_{i}=\sum_{k} X_{i k}$ be the number of times any word $k$ appears in the context of word $i$. Let $P_{i j}=P\left(w_{j} \lvert w_{i}\right)=\frac{X_{i j}}{X_{i}}$ be the probability of $j$ appearing in the context of word $i$.


$$
\begin{array} { l | l l l l l } 
{ \text { Probability and Ratio } } & { k = \text { solid } } & { k = g a s } & { k = \text { water } } & { k = \text { fashion } } \\
\hline P(k | i c e)& {1.9 \times 10^{-4}} & {6.6 \times 10^{-5}} & {3.0 \times 10^{-3}} & {1.7 \times 10^{-5}} \\
{P(k | \text {steam)}} & {2.2 \times 10^{-5}} & {7.8 \times 10^{-4}} & {2.2 \times 10^{-3}} & {1.8 \times 10^{-5}} \\
{P(k | i c e) / P(k |\text {steam)} } & {8.9} & {8.5 \times 10^{-2}} & {1.36} & {0.96}
\end{array}
$$


Here is an example in original paper. The relationship of word *ice* and *steam* can be examined by studying the ratio of their co-occurrence probabilities with various probe words, *k*. For words *k* related to ice but not steam, say *k* = *solid*, we expect the ratio $P_{ik}/P_{jk}$ will be large. Similarly, for words *k* related to steam but not ice, say *k* = *gas*, the ratio should be small. For words *k* like *water* or *fashion*, that are either related to both ice and steam, or to neither, the ratio should be close to one. 

Now, how can we capture ratios of co-occurrence probabilities as linear meaning components in a word vector space. The most natural way to do this is with vector differences. Suppose we have word embedding $w_{i},w_{j},\tilde{w}_{k}$, where  $w \in \mathbb{R}^{d}$ and hat $\tilde{}$ represent context word, we want to find $F$ that:


$$
F\left(w_{i}-w_{j}, \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}}
$$


Notice that the left of equation is vector but right is a scale. One simple and intuitive solution for this is dot product:


$$
F\left((w_{i}-w_{j})^T\tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}}
$$


Further, we can assume follow:


$$
F\left((w_{i}-w_{j})^T\tilde{w}_{k}\right)=\frac{F(w_{i}\tilde{w_k})}{F(w_{j}\tilde{w_k})}=\frac{P_{i k}}{P_{j k}}
$$


The intuitive solution of $F$ would be exponential, equally:


$$
w_{i}^{T} \tilde{w}_{k}=\log \left(P_{i k}\right)=\log \left(X_{i k}\right)-\log \left(X_{i}\right)\\
(w_{i}-w_{j})^T\tilde{w}_{k}=log(\frac{P_{ik}}{P_{jk}})
$$


Finally, notice that $w$ and $\tilde{w}$ are actually exchangable, that is


$$
\tilde{w}_{i}^{T} w_{k}=\log \left(P_{i k}\right)=\log \left(X_{i k}\right)-\log \left(X_{i}\right)
$$


also hold.  So far, we only have:


$$
w_{k}^{T} \tilde{w}_{i}=\log \left(P_{k i}\right)=\log \left(X_{k i}\right)-\log \left(X_{k}\right)
$$


The $X_{ik}$ and $X_{ki}$ are equal in co-occurence matrix, the only difference is between $\log \left(X_{i}\right)$ and $\log \left(X_{k}\right)$. But we can see these two terms are independent from each other,  thus we can use bias to replace them:


$$
w_{i}^{T} \tilde{w}_{k}+b_{i}+\tilde{b}_{k}=\log \left(X_{i k}\right)
$$


The final loss function is:


$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}
$$


Where $f$ is weight term used to characterize the frequence of every word. This is also sort of hyper-parameter we can choose, In original paper, the $f$ is:


$$
f(x)=\left\{\begin{array}{cc}
{\left(x / x_{\max }\right)^{\alpha}} & {\text { if } x<x_{\max }} \\
{1} & {\text { otherwise }}
\end{array}\right.
$$


and $x_{max}$ is 100, $\alpha=\frac{3}{4}$.

#### Relationship to Skip-Gram

Recall that in Skip-Gram model, our loss function is:


$$
J=-\sum_{i \in \text {corpus}} \sum_{j \in \text {context}(i)} \log Q_{i j}
$$


Where $Q_{ij}$ is softmax probability. As the same words $i$ and $j$ can appear multiple times in the corpus, it is more efficient to first group together the same values for $i$ and $j$:


$$
J=-\sum_{i=1}^{V} \sum_{j=1}^{V} X_{i j} \log Q_{i j}
$$


Use $P_{i j}=\frac{X_{i j}}{X_{i}}$, we can replace $X_{ij}$ as:


$$
J=-\sum_{i=1}^{V} X_{i} \sum_{j=1}^{V} P_{i j} \log Q_{i j}=\sum_{i=1}^{V} X_{i} H\left(P_{i}, Q_{i}\right)
$$


$H(P_{i},Q_{i})$ is cross-entropy characterize the difference between estimated distribution $Q_{i}$ and co-occurence distribution $P_{i}$. However, a big drawback of this is that $Q_{i}$ need to be normalized, which is computational expensive. A solution to this is least square error estimation:


$$
\hat{J}=\sum_{i=1}^{V} \sum_{j=1}^{V} X_{i}\left(\hat{P}_{i j}-\hat{Q}_{i j}\right)^{2}
$$


Where $$\hat{P}_{i j}=X_{i j}$$ and $$\hat{Q}_{i j}=\exp (\vec{v}_{j}^{T} \vec{v}_{i}+b_{i}+\tilde{b_j})$$. Finally, take log and use same $f$ in GloVe, we get:


$$
\hat{J}=\sum_{i, j} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b_j}-\log X_{i j}\right)^{2}
$$


Which is the same as what we saw in GloVe.

GloVe turns out to be computational efficiently by only training on the nonzero elements in co-occurrence matrix. Meanwhile, it can produces a vector space capture not only similarity between words, but sub-structure behind words.



### Reference

[1] CS 224N Lecture notes, Standford. http://web.stanford.edu/class/cs224n/index.html

[2] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.GloVe: Global Vectors for Word Representation.

