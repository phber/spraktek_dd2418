
from sklearn.metrics.pairwise import euclidean_distances
dist = euclidean_distances(x.toarray().T)
vocab = v.get_feature_names()

import matplotlib.pyplot as plt

from sklearn.manifold import MDS

print 'Plotting'
# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
for x, y, name in zip(xs, ys, vocab):
    plt.scatter(x, y)
    plt.text(x, y, name)

plt.show()
