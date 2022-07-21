# Dirichlet Mixture Processes for Clustering

## Introduction 

Cluster analysis is the task of partitioning a set of observations into sub-groups, called a clusters, such that observations in the same clusters are closer with respect to a similarity measure to each other than to those in other clusters. In theory, data points that are in the same group (cluster) should have high similarity, while data points in different groups should have lower similarity. The choice of the similarity measure is dependent on the specific problem. An example of clustering is shown in Figure 1, where we show how unlabelled data can be labelled according to three clusters.


<figure>
<p align="center">
    <img src="https://user-images.githubusercontent.com/93731561/180188827-b4e64d74-03fd-4898-bca0-8f6fbca35c38.png" width=70% height=70%>
    <figcaption> Figure 1: Clustering exmaple, image taken from [1]</figcaption>
</p>
</figure>
<br/>

There are different methods for performing clustering, where the most famous ones are KMeans and Mixture Models. Specifically, Micture Models account for variance and returns the probability that a data point belongs to a specific cluster, contrarly to Kmeans that only assign a cluster to a specific datapoint. Nevertheless, both algorithms require the number of cluster to be known at-priori, which is not always guaranteed. An exaustive search can be conducted for finding the optimal number of clusters, but if the dimension of the problem is really big this becomes unfeasible. We would like a way to infer directly the number of clusters from data. Furthemore, we would like to have a variable number of clusters in order to handle novelty detection, hence as data increases clusters (should) increase. Both of these two problems can be solved by the use of Dirichlet Mixture Models.

## Dirichlet Processes

A dirichlet process is a stochastic process that takes values over probability distributions, hence it can be seen as a distribution over distributions. They belong to the class of no-parametric Bayesian models, if you are interested consider reading [2]. A nice introduction to Dirichlet Processes is given in this [Medium](https://medium.com/@albertoarrigoni/dirichlet-processes-917f376b02d2) article. Mathematically we define a Dirchlet Process as follow:

*A **Dirichlet Process (DP)** is a stochastic process that takes values over probability distributions* $G : \Theta \rightarrow \mathbb{R}^+$, *such that*:
<p align="center">
$G(\theta) \geq 0 \quad \forall\,\theta\in \Theta \\ \int_\Theta G(\theta) d\theta = 1$
</p>

*For each partition* $T_1, \dots, T_k$ of $\Theta$, *a DP is defined implicitly by the requirement*:  
<p align="center">
    $(G(T_1), \dots, G(T_K)) \sim Dir(H(\alpha T_1), \dots, H(\alpha T_K))$
</p>

*where* $\alpha$ *is the concentration parameter,* $H$ *is the base measure and* $Dir(-)$ *indicates the Dirichlet distribution*.

We consider dirichlet processes applied to mixture models, in particular we will use the following Probabilistic Graphical Model (PGM):

<br/>
<figure>
<p align="center">
    <img src="https://user-images.githubusercontent.com/93731561/180213604-8dfa87ae-3040-490a-9fbd-7ee13f68374d.png" width=20% height=20%>
    <figcaption> Figure 2: PMG for DP mixtures, image taken from [3]</figcaption>
</p>
</figure>
<br/>
Each observation $x_i$ is sampled from a probability distribution $F$, which depends on $\bar{\theta_i}$ parameters. These parameters are sampled from a $G$ which is a Dirichlet Process of concentration parameter $\alpha$ and base measure $H$. Notice that the we have two hyperparameters in the model $\alpha$ and $\lambda$, that represent our prior belief of the system. Furthemore, each $x_i$ might be sampled in theory from a different probability distribution, since the $\bar{\theta_i}$ might be different for different observations. In practice, the sampling from $G$ is done in different way, here we present the stick-breaking mechanism (GEM). The full model is: 


$$\begin{align*}
        &\pi \sim \mathop{GEM}(\alpha) & 
        \bar{\theta_i} & \sim G(\theta) =\sum_{k=1}^\infty \pi_k \delta_{\theta_k}(\theta) \\
        &\theta_k \sim H(\lambda) &
        \mathbf{x}_i &  \sim F(\bar{\theta}_i)
\end{align*}$$

Where the $GEM$ process is:

$$    
\begin{align*}
        \beta_k & \sim Beta(1, \alpha) \\ 
        \pi_k & = \beta_k \prod_{l=1}^{k-1}(1 - \beta_l) =  \beta_k \Big(1 - \sum_{l=1}^{k-1}\pi_l\Big)
\end{align*}
$$

As a consequence of this construction, samples from a DP are discrete with probability one, which means that if you keep sampling it, more and more repetitions of previously generated values are going to be obtained. This is indeed a possible way to perform clustering. Notice that $(\pi_k)_{k\geq1}$
represent the weights of the mixture, possibly infinite. In practice, we can choose a very high bound to perform calculation, e.g. choosing the number of data point. This will be reflected as the bound on the number of possible clusters.

## Implementation

The model we use consider $F$ as a Normal distribution. We use `sklearn.mixture.BayesianGaussianMixture` which implements the model we described above and perform learning by the use of Variational Inference. The model is first tested on a synthetic dataset made by mutimodal gaussians and the it is used for clustering the _20newgroups dataset_. Due to the high dimensionality of the data after the preprocessing, done by means of tf-idf, we used a dimensionality reduction technique, named t-SNE. This is a stochastic dimensionality reduction technique very useful to visualize high dimensional sparse data, cosider reading [4].

## Results

A small presentation done by me and my colleague [Alessandro Pierro]() at the University of Trieste during our Master Course is available to check results and for further details.

## References

1. F. Ebadi and M. Norouzi, "Road Terrain detection and Classification algorithm based on the Color Feature extraction," 2017 Artificial Intelligence and Robotics (IRANOPEN), 2017, pp. 139-146, doi: 10.1109/RIOS.2017.7956457.
2. Hjort, N., Holmes, C., Müller, P., & Walker, S. (Eds.). (2010). Bayesian Nonparametrics (Cambridge Series in Statistical and Probabilistic Mathematics). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511802478
3. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. Cambridge, MA: MIT Press.
4. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(86), 2579–2605.
