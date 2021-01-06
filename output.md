# Clustering Algorithms Comparison on Wine Data

#### Compared Algorithms:
    *
K-Means Clustering
    * Agglomerative Clustering
    * DBSCAN Clustering
    *
Mean-Shift Clustering
    * BIRCH Clustering
    * Affinity Propagation
    *
Mini-batch k-means
    * Spectral Clustering
    
## Objective
   _Clustering
aims to maximize intra-cluster similarity and minimize inter-cluster
similarity._

    Each clustering problems requires own unique solutions.
According to my observation, most of tutorials and guidebooks focus on K-means
clustering and the data preparation process before. I want to introduce other
clustering algorithms and  to inform when do we need other algorithms. 
    
##
Data
   Due to more practical explanation, I am going to use
   [Wine
Dataset](https://www.kaggle.com/harrywang/wine-dataset-for-clustering)

## Review of the Data

```python
!pip install -U scikit-learn
```

```python
import sklearn
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
```

```python
data = pd.read_csv(r'C:\Users\Arda\Downloads\archive (2)/wine-clustering1.csv')

data.head()
```

```python
data.dtypes
```

```python
data.isnull().sum()
```

```python
data.describe()


```

```python
sns.set(style='white',font_scale=1.3, rc={'figure.figsize':(20,20)})
ax=data.hist(bins=20,color='blue')
```

Some of our features distributed normally some of not, it is natural. Most
important insight in the figure above is there are not well balanced red and
white wine distribution. Number of white wines almost 3 times of red wines. This
is an essential point for algorithm performance.

```python
data.plot( kind = 'box', subplots = True, layout = (4,6), sharex = False, sharey = False,color='blue')
plt.show()
```

```python
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True, linewidths=5, ax=ax)
plt.show()
```

```python

sns.pairplot(data)
```

As you can see, there are lots of similar features and noise in the data. We
should apply dimension reduction techniques for
well selected features for
clustering algorithms.

## Dimension Reduction

## Clustering Algorithms
### 1) Centroid Based
        Cluster represented by
central reference vector which may not be a part of the original data e.g
k-means clustering
        
        * K-means Clustering
### 2) Hierarchical
Connectivity based clustering based on the core idea that points are connected
to points close by rather than 
        further away. A cluster can be defined
largely by the maximum distance needed to connect different parts of the
cluster. Algorithms do not partition the dataset but instead construct a tree of
points which are typically 
        merged together.
        
        *
Agglomerative Clustering
        * BIRCH Clustering
### 3) Distribution Based
Built on statistical distribution models - objects of a cluster are the ones
which belong likely to same 
        distribution. Tend to be complex clustering
models which might be prone to overfitting on data points
        
        *
Gausssian mixture models
### 4) Density Based
        Create clusters from areas
which have a higher density of data points. Objects in sparse areas, which
seperate 
        clusters, are considered noise and border points.
* DBSCAN Clustering
        * Mean-shift Clustering

![2021-01-06_23-38-57.png](attachment:2021-01-06_23-38-57.png)

```python
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
```

```python
wine_df = pd.read_csv('C:/Users/Arda/Downloads/archive (2)/wine-clustering1.csv')


```

```python
wine_df.shape
```

```python
wine_df = wine_df.sample(frac=1).reset_index(drop=True)
wine_df.head()
wine_df.info()
```

```python
wine_features = wine_df.drop('Color_Intensity', axis=1)
wine_features.head()
```

```python
is_wine_red_or_white = wine_df['Color_Intensity']
is_wine_red_or_white.sample(5)
```

### Evaluation Metrics
##### Homogeneity Score
    Clustering satisfies
homogeneity if all of its clusters contains only points which are members of a
single class.
    The actual label values do not matter i.e the fact that actual
label 1 corresponds to cluster label 2 does
    not affect this score
#####
Completeness Score
    Clustering satisfies completeness if all the points that
are members of the same class belong to the same cluster
##### V Measure Score
Harmonic mean of homogeneity and completeness score - usually used to find the
avarage of rates
##### Adjusted Rand Score
    Similarity measure between
clusters which is adjusted for chance i.e random labeling of data points
Close to 0: data was randomly labeled
    Exact 1: actual and predicted clusters
are identical 
##### Adjusted Mutual Information Score
    Information obtained
about one random variable by observing another random variable adjusted to
account for chance
    Close to 0: data was randomly labeled
    Exact 1: actual
and predicted clusters are identical
##### Silhouette Score
    Uses a distance
metric to measure how similar a point is to its own cluster and how dissimilar
the point is from
    points in other clusters. Ranges between -1 and 1 and
positive values closer to 1 indicate that the clustering
    was good

```python
def BuildModel(clustering_model,data,labels):
    model=clustering_model(data)
    print('homo\tcompl\tv-means\tARI\tAMI\tsilhouette')
    print(50*'_')
    print('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
         %(metrics.homogeneity_score(labels, model.labels_),
           metrics.completeness_score(labels, model.labels_),
           metrics.v_measure_score(labels, model.labels_),
           metrics.adjusted_rand_score(labels, model.labels_),
           metrics.adjusted_mutual_info_score(labels, model.labels_),
           metrics.silhouette_score(data,model.labels_)))
```

#### K-Means Clustering
    To process the learning data, the K-means algorithm
in data mining starts with a first group of randomly
    selected centroids,
which are used as the beginning points for every cluster, and then performs
iterative (repetitive) 
    calculations to optimize the positions of the
centroids.It halts creating and optimizing clusters when either:
    The
centroids have stabilized — there is no change in their values because the
clustering has been successful.
    The defined number of iterations has been
achieved.
    
    
### Contrasting K-Means and Hierarchical Clustering
#####
K-Means
    * Need distance measure as well as way to aggregate points in a
cluster
    * Must represent data as vectors in N-dimensional hyperspace
    *
Data representation can be difficult for complex data types
    * Variants can
efficiently deal with very large datasets on disk
##### Hierarchical
    * Only
need distance measure; do not need way to combine points in cluster
    * No
need to express data as vectors in N-dimensional hyperspace
    * Relatively
simple to represent even complex documents
    * Even with careful construction
too computationaly expensive for large datasets on disk

```python
def k_means(data,n_clusters=2, max_iter=1000):
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(data)
    
    return model
```

```python
BuildModel(k_means,wine_df,is_wine_red_or_white)
```

#### According to scores our k-means cluster perform did not well. Let's try
other algorithms and hope an accuracy improvement



```python

```
