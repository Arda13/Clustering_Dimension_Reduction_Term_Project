
# Clustering Algorithms Comparison on Wine Data

#### Compared Algorithms:
    * K-Means Clustering
    * Agglomerative Clustering
    * DBSCAN Clustering
    * Mean-Shift Clustering
    * BIRCH Clustering
    * Affinity Propagation
    * Mini-batch k-means
    * Spectral Clustering
    
## Objective
   _Clustering aims to maximize intra-cluster similarity and minimize inter-cluster similarity._

    Each clustering problems requires own unique solutions. According to my observation, most of tutorials and guidebooks focus on K-means clustering and the data preparation process before. I want to introduce other clustering algorithms and  to inform when do we need other algorithms. 
    
## Data
   Due to more practical explanation, I am going to use
   [Wine Dataset](https://www.kaggle.com/harrywang/wine-dataset-for-clustering)

## Review of the Data


```python
!pip install -U scikit-learn
```

    Requirement already up-to-date: scikit-learn in c:\users\arda\anaconda3\lib\site-packages (0.24.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in c:\users\arda\anaconda3\lib\site-packages (from scikit-learn) (1.2.1)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in c:\users\arda\anaconda3\lib\site-packages (from scikit-learn) (2.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in c:\users\arda\anaconda3\lib\site-packages (from scikit-learn) (0.13.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.13.3 in c:\users\arda\anaconda3\lib\site-packages (from scikit-learn) (1.16.4)
    


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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>0</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>0</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>0</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>1</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>0</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dtypes
```




    Alcohol                 float64
    Malic_Acid              float64
    Ash                     float64
    Ash_Alcanity            float64
    Magnesium                 int64
    Total_Phenols           float64
    Flavanoids              float64
    Nonflavanoid_Phenols    float64
    Proanthocyanins         float64
    Color_Intensity           int64
    Hue                     float64
    OD280                   float64
    Proline                   int64
    dtype: object




```python
data.isnull().sum()
```




    Alcohol                 0
    Malic_Acid              0
    Ash                     0
    Ash_Alcanity            0
    Magnesium               0
    Total_Phenols           0
    Flavanoids              0
    Nonflavanoid_Phenols    0
    Proanthocyanins         0
    Color_Intensity         0
    Hue                     0
    OD280                   0
    Proline                 0
    dtype: int64




```python
data.describe()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Color_Intensity</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.591124</td>
      <td>0.280899</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572394</td>
      <td>0.450706</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>0.000000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>0.000000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>0.000000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>1.000000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>1.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style='white',font_scale=1.3, rc={'figure.figsize':(20,20)})
ax=data.hist(bins=20,color='blue')
```


![png](output_8_0.png)


Some of our features distributed normally some of not, it is natural. Most important insight in the figure above is there are not well balanced red and white wine distribution. Number of white wines almost 3 times of red wines. This is an essential point for algorithm performance.


```python
data.plot( kind = 'box', subplots = True, layout = (4,6), sharex = False, sharey = False,color='blue')
plt.show()
```


![png](output_10_0.png)



```python
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True, linewidths=5, ax=ax)
plt.show()
```


![png](output_11_0.png)



```python

sns.pairplot(data)
```




    <seaborn.axisgrid.PairGrid at 0x1c2e8a41cf8>




![png](output_12_1.png)


As you can see, there are lots of similar features and noise in the data. We should apply dimension reduction techniques for
well selected features for clustering algorithms. 

## Dimension Reduction

## Clustering Algorithms
### 1) Centroid Based
        Cluster represented by central reference vector which may not be a part of the original data e.g k-means clustering
        
        * K-means Clustering
### 2) Hierarchical 
        Connectivity based clustering based on the core idea that points are connected to points close by rather than 
        further away. A cluster can be defined largely by the maximum distance needed to connect different parts of the
        cluster. Algorithms do not partition the dataset but instead construct a tree of points which are typically 
        merged together.
        
        * Agglomerative Clustering
        * BIRCH Clustering
### 3) Distribution Based
        Built on statistical distribution models - objects of a cluster are the ones which belong likely to same 
        distribution. Tend to be complex clustering models which might be prone to overfitting on data points
        
        * Gausssian mixture models
### 4) Density Based
        Create clusters from areas which have a higher density of data points. Objects in sparse areas, which seperate 
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




    (178, 13)




```python
wine_df = wine_df.sample(frac=1).reset_index(drop=True)
wine_df.head()
wine_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 13 columns):
    Alcohol                 178 non-null float64
    Malic_Acid              178 non-null float64
    Ash                     178 non-null float64
    Ash_Alcanity            178 non-null float64
    Magnesium               178 non-null int64
    Total_Phenols           178 non-null float64
    Flavanoids              178 non-null float64
    Nonflavanoid_Phenols    178 non-null float64
    Proanthocyanins         178 non-null float64
    Color_Intensity         178 non-null int64
    Hue                     178 non-null float64
    OD280                   178 non-null float64
    Proline                 178 non-null int64
    dtypes: float64(10), int64(3)
    memory usage: 18.2 KB
    


```python
wine_features = wine_df.drop('Color_Intensity', axis=1)
wine_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol</th>
      <th>Malic_Acid</th>
      <th>Ash</th>
      <th>Ash_Alcanity</th>
      <th>Magnesium</th>
      <th>Total_Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid_Phenols</th>
      <th>Proanthocyanins</th>
      <th>Hue</th>
      <th>OD280</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.25</td>
      <td>4.72</td>
      <td>2.54</td>
      <td>21.0</td>
      <td>89</td>
      <td>1.38</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>0.80</td>
      <td>0.75</td>
      <td>1.27</td>
      <td>720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.05</td>
      <td>5.80</td>
      <td>2.13</td>
      <td>21.5</td>
      <td>86</td>
      <td>2.62</td>
      <td>2.65</td>
      <td>0.30</td>
      <td>2.01</td>
      <td>0.73</td>
      <td>3.10</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.84</td>
      <td>2.89</td>
      <td>2.23</td>
      <td>18.0</td>
      <td>112</td>
      <td>1.72</td>
      <td>1.32</td>
      <td>0.43</td>
      <td>0.95</td>
      <td>0.96</td>
      <td>2.52</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.08</td>
      <td>1.83</td>
      <td>2.32</td>
      <td>18.5</td>
      <td>81</td>
      <td>1.60</td>
      <td>1.50</td>
      <td>0.52</td>
      <td>1.64</td>
      <td>1.08</td>
      <td>2.27</td>
      <td>480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.68</td>
      <td>1.83</td>
      <td>2.36</td>
      <td>17.2</td>
      <td>104</td>
      <td>2.42</td>
      <td>2.69</td>
      <td>0.42</td>
      <td>1.97</td>
      <td>1.23</td>
      <td>2.87</td>
      <td>990</td>
    </tr>
  </tbody>
</table>
</div>




```python
is_wine_red_or_white = wine_df['Color_Intensity']
is_wine_red_or_white.sample(5)
```




    91     0
    162    0
    132    0
    160    0
    78     0
    Name: Color_Intensity, dtype: int64



### Evaluation Metrics
##### Homogeneity Score
    Clustering satisfies homogeneity if all of its clusters contains only points which are members of a single class.
    The actual label values do not matter i.e the fact that actual label 1 corresponds to cluster label 2 does
    not affect this score
##### Completeness Score
    Clustering satisfies completeness if all the points that are members of the same class belong to the same cluster
##### V Measure Score
    Harmonic mean of homogeneity and completeness score - usually used to find the avarage of rates
##### Adjusted Rand Score
    Similarity measure between clusters which is adjusted for chance i.e random labeling of data points
    Close to 0: data was randomly labeled
    Exact 1: actual and predicted clusters are identical 
##### Adjusted Mutual Information Score
    Information obtained about one random variable by observing another random variable adjusted to account for chance
    Close to 0: data was randomly labeled
    Exact 1: actual and predicted clusters are identical
##### Silhouette Score
    Uses a distance metric to measure how similar a point is to its own cluster and how dissimilar the point is from
    points in other clusters. Ranges between -1 and 1 and positive values closer to 1 indicate that the clustering
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
    To process the learning data, the K-means algorithm in data mining starts with a first group of randomly
    selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) 
    calculations to optimize the positions of the centroids.It halts creating and optimizing clusters when either:
    The centroids have stabilized — there is no change in their values because the clustering has been successful.
    The defined number of iterations has been achieved.
    
    
### Contrasting K-Means and Hierarchical Clustering
##### K-Means
    * Need distance measure as well as way to aggregate points in a cluster
    * Must represent data as vectors in N-dimensional hyperspace
    * Data representation can be difficult for complex data types
    * Variants can efficiently deal with very large datasets on disk
##### Hierarchical
    * Only need distance measure; do not need way to combine points in cluster
    * No need to express data as vectors in N-dimensional hyperspace
    * Relatively simple to represent even complex documents
    * Even with careful construction too computationaly expensive for large datasets on disk


```python
def k_means(data,n_clusters=2, max_iter=1000):
    model = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(data)
    
    return model
```


```python
BuildModel(k_means,wine_df,is_wine_red_or_white)
```

    homo	compl	v-means	ARI	AMI	silhouette
    __________________________________________________
    0.012	0.012	0.012	0.042	0.008	0.657
    

#### According to scores our k-means cluster perform did not well. Let's try other algorithms and hope an accuracy improvement




```python

```
