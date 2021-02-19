---
title: "Random Forest"
categories:
  - Machine Learning
tags:
  - Learning note
classes: wide

---



### 1.Bagging

Beside boosting, bagging is another ensemble learning technique.  The full name of bagging is called **Bootstrapped Aggregating**. The main idea behind bagging is that we use bootsrapping to generate different datasets. and we train a weak learner on each dataset, finally, we aggregate each weak learner to generate a strong learner. In bagging, we usually use unweighted aggregation:


$$
G(x)=\bar{g}(x)=\operatorname{sign}\left(\frac{1}{M} \sum_{m=1}^{M} g_{m}(x)\right)
$$


Where $g_m$ is weak learner, $G$ is strong learner. So, why is bagging might helpful, we can see this from statistical way. Consider $M$ independent random variables $x_{1}, x_{2}, \ldots, x_{M}$ with variance $\sigma^2$, thus:


$$
var(\frac{1}{M} \sum_{m=1}^{M} x_{m})=\frac{\sigma^2}{M}
$$


As a result, if we have weak learners that have high variance and low bias(oppoiste to adaBoost ), bagging can help to reduce the variance but mentain low bias.

### 2. Bootstrapping

Bootstrapping is a resampling technique which is principally used to estimate various measures of error or undcertainty of parameter estimates. The procedure of bootstrapping is follow:

* let $D=\left\{\left(x_{1}, y_{1}\right), \ldots,\left(x_{N},  y_{N}\right)\right\}$ be the dataset we have, the number of $D$ is $N$
* We repeatedly uniformly sample $N$ points from $D$ **with replacement**
* The new dataset is called bootstrapping dataset.

We can see that, the probability for a sample not be in the particular bootstrapping dataset is:


$$
(1-\frac{1}{N})^N
$$


If $N \rightarrow\infty$ , we can see the value of above would approach to 0.368, which means that for every bootstrapping dataset, there are about 36.8% sample would not be choosen. We call thest samples **Out-of Bag** samples.



### 3. Random Forest

#### 3.1 Algorithm

**Random Forest** is one of the most popular bagging algorithm. The main idea of random forest is to use decision tree as weak learner. However, unlike adaBoost, there we usually use max-depth tree.  The procedure of random forest is follow:

* construct several bootstrapping dataset.
* for each dataset $m$, we learn a decision tree model $g_m$
* aggregate result for all the weak learner
  * for classification:$\bar{g}(x)=\operatorname{sign}\left(\frac{1}{M} \sum_{m=1}^{M} g_{m}(x)\right)$
  * for regression: $\bar{g}(x)=\frac{1}{M} \sum_{m=1}^{M} g_{m}(x)$

 In practical, we not only randomly select dataset, but also randomly select subset for features for each dataset to increase the randomization. 



#### 3.2 Out-of-bag error

As we mentioned, for each bootstrapping dataset, there are about 36.8% data would not be included. Thus, we can use these samples to do validation.

<img src="/assets/images/image-20200331102358985.png" alt="image-20200331102358985" style="zoom:33%;" />

Then, for each data sample, we have:

* $G_1^-=aggregate(g_3,g_4,...)$
* $G_2^-=aggregate(g_2,g_3,g_4,...)$
* $G_n^-=aggregate(g_1,...)$

Finally, we can define **Out-of-bag error** as follow:


$$
E_{O O B}(G)=\frac{1}{N} \sum_{n=1}^{N} \operatorname{error}\left(G_{n}^{-}\left(\vec{x}_{n}\right), y_{n}\right)
$$


which is intrinsic mechanism for us to perform validation. Thus, in random forest, we actually don't need to split dataset to train and test set.



#### 3.3 Feature Importance

Finally, we discuss how we can calculate feature importance in random forest. There are basically two ways to do this. the one is based on information gain, another is use out-of-bag data.

For information gain, we notice that, for each weak learner $g_m$, we can calculate the information gain for feature $i$ in node $j$. Then, the importance for feature $i$ in $g_m$ would be avarage the information gain for all nodes that use feature $i$. Finally, we avarage the score for all $g_m$ to get feature importance score for feature $i$.

For out-of-bag data, the main idea is that, if a feature is important, then, if we change the value for this feature, the prediction result would change a lot. Instead, if feature is not important, change the value of this feature would not change the prediction result to much. We use out-of-bag data to do this:


$$
v=\frac{n_{y=y^{*}}-n_{y=y_{\pi}^*}}{|o o b|}
$$


where $v$ is the importance for feature $i$ in weak learner $g_m$, $n_{y=y^{*}}$ is the number of sample that have correct prediction before we change the value of feature. $n_{y=y_{\pi}^*}$ is the number of sample that have correct prediction after we change the value of feature. $|oob|$ is the number of out-of-bag sample in this dataset. Finally, we can avarage $v$ to get final importance score.



### Reference

[1] Lectures of CSE417T in Washington University in St.Louis, Chien-Ju Ho.http://chienjuho.com/courses/cse417t/

[2] tensorinfinity. http://www.tensorinfinity.com

