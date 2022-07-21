from matplotlib.patches import Ellipse
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.datasets import load_digits
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# === Dataset extraction === #
categories = [
    "alt.atheism",
    "rec.sport.baseball",
    "talk.politics.mideast",
    "sci.space",
]

news_groups_train = fetch_20newsgroups(
    subset='train', shuffle=True, download_if_missing=False, categories=categories)
news_groups_test = fetch_20newsgroups(
    subset='test', shuffle=True, download_if_missing=False, categories=categories)
x_train, y_train = news_groups_train.data, news_groups_train.target
x_sp_train, x_sp_val, y_sp_train, y_sp_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=0)
x_test, y_test = news_groups_test.data, news_groups_test.target


# refer: http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
tfidf_vectorizer = TfidfVectorizer(max_df=0.97, min_df=2,
                                   max_features=5000,
                                   stop_words='english')
t0 = time.time()
tfidf_x_sp_train = tfidf_vectorizer.fit_transform(x_sp_train)
tfidf_x_sp_val = tfidf_vectorizer.transform(x_sp_val)

tfidf_x_train = tfidf_vectorizer.transform(x_train)
tfidf_x_test = tfidf_vectorizer.transform(x_test)
print("done in %0.3fs." % (time.time() - t0))


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40,
            n_iter=10000, init='random')
tsne_results = tsne.fit_transform(tfidf_x_sp_train)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    palette=sns.color_palette("hls", 4),
    hue=y_sp_train,
    legend=None,
    alpha=0.3
)


# === Bayesian Dirichlet Mixture === #
dpgmm = mixture.BayesianGaussianMixture(
    n_components=20, covariance_type="full", max_iter=1000,
    weight_concentration_prior_type='dirichlet_distribution',
    init_params="random",
    mean_precision_prior=6,
    weight_concentration_prior=0.00000001,
    n_init=2, random_state=2).fit(tsne_results)

print("Final number of active components: %d" % np.sum(dpgmm.weights_ > 0.01))

preds = dpgmm.predict(tsne_results)
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    palette=sns.color_palette("hls", len(np.unique(preds))),
    hue=preds,
    legend="full",
    alpha=0.3
)
len(np.unique(preds))


def plot_cov_ellipse(cov, pos, col='b', nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height,
                    angle=theta, color=col, alpha=0.2, **kwargs)
    ax.add_artist(ellip)
    return ellip


colors = sns.color_palette("hls", len(np.unique(preds)))
plt.figure(figsize=(16, 10))
for n, color in zip(np.unique(preds), colors):

    data = tsne_results
    if dpgmm.covariance_type == 'full':
        cov = dpgmm.covariances_[n]
    elif dpgmm.covariance_type == 'tied':
        cov = dpgmm.covariances_[n]
    elif dpgmm.covariance_type == 'diag':
        cov = np.diag(dpgmm.covariances_[n])
    elif dpgmm.covariance_type == 'spherical':
        cov = np.eye(dpgmm.means_.shape[1]) * dpgmm.covariances_[n]
    # print(cov)
    pos = dpgmm.means_[n]
    # print(pos)
    plot_cov_ellipse(cov, pos, col=color)

sns.scatterplot(
    x=data[:, 0], y=data[:, 1],
    hue=preds,
    palette=colors,
    legend=None,
    alpha=0.3
)
plt.show()
