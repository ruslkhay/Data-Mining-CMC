# Comprehensive Analysis of Clustering and Anomaly Detection Techniques

This notebook implements various clustering and anomaly detection techniques on baseball player data and handwritten digit images. 

## Sub-task 1: Data Loading

The notebook loads baseball player data from a CSV file. This dataset contains 
performance statistics, playing time, league information, and salary data for 
baseball players. The `Name` column is treated as the record identifier.

## Sub-task 2: Handling Missing Values

### KNNImputer Algorithm
The notebook uses `KNNImputer` with k=3 neighbors to handle missing values, particularly in the Salary variable. 

**How KNNImputer works:**
1. For each sample with missing values, it identifies the k-nearest neighbors based on the features that are not missing
2. It then imputes the missing values using the weighted average of the corresponding feature values from the k-nearest neighbors
3. The weight is proportional to the inverse of the distance to the neighbor

**Why use KNNImputer:**
- It preserves the relationships between features since it uses information from similar data points
- It's more sophisticated than simple mean/median imputation as it considers the local structure of the data
- With k=3, it balances between stability (higher k) and locality (lower k)

After imputation, the notebook recalculates `logSalary` as `log(1+Salary)`. The logarithmic transformation is applied to:
- Make the salary distribution more symmetrical
- Reduce the influence of very high salaries (outliers)
- The 1+ is added to handle potential zero values (since log(0) is undefined)

## Sub-task 3: Variable Normalization

### MinMaxScaler
**What it does:** Normalizes numeric features to a [0,1] range using the formula: `X_scaled = (X - X_min) / (X_max - X_min)`

**Why use it:**
- Ensures all numeric features have similar scales, preventing features with larger ranges from dominating the distance calculations
- Essential for distance-based clustering methods where scale matters
- Unlike StandardScaler, it produces bounded values, which can be beneficial for some algorithms

### OneHotEncoder
**What it does:** Converts categorical variables into binary vectors (one column per category)

**Why use it:**
- Many algorithms can't directly work with categorical data
- One-hot encoding creates a binary column for each category
- Prevents algorithms from interpreting categorical variables as having numerical relationships (e.g., "American League" isn't greater than "National League")

Together, these preprocessing steps ensure that all features contribute appropriately to the clustering process, regardless of their original scale or type.

## Sub-task 4: Hierarchical Clustering

The notebook implements bottom-up (agglomerative) hierarchical clustering with:
- **Link method**: "complete" (maximum or furthest-neighbor linkage)
- **Distance metric**: "manhattan" (also called cityblock or L1 distance)

### Agglomerative Hierarchical Clustering Algorithm
**How it works:**
1. Initially, each data point is its own cluster
2. Iteratively merge the two closest clusters based on the specified distance and linkage criteria
3. Continue until all points are in a single cluster

**Complete Linkage**: The distance between two clusters is defined as the maximum distance between any point in the first cluster and any point in the second cluster. This tends to create more compact, equally-sized clusters.

**Manhattan Distance**: The sum of absolute differences between coordinates. Unlike Euclidean distance, it doesn't square the differences, making it less sensitive to outliers.

**Why use these settings:**
- Complete linkage prevents the "chaining effect" seen in single linkage where clusters can be elongated
- Manhattan distance is appropriate when features are on different scales or when outliers might be present
- This combination is particularly good for finding compact, well-separated clusters

The **dendrogram** visualizes the hierarchical structure, showing the order of merges and the distances at which they occur. The truncation to the top 20 clusters helps make the visualization more interpretable.

## Sub-task 5: Finding Optimal Number of Clusters

The notebook uses the **Pseudo-F criterion** (also known as the Calinski-Harabasz index) to determine the optimal number of clusters.

### Pseudo-F Criterion
**How it works:**
1. Calculate the ratio of between-cluster variance to within-cluster variance
2. Higher values indicate better clustering (well-separated, compact clusters)
3. A local peak in the pseudo-F value suggests a good number of clusters

The formula is:
$$\text{Pseudo-F} = \frac{\text{SSB} / (k-1)}{\text{SSW} / (n-k)}$$

Where:
- SSB is the sum of squares between clusters
- SSW is the sum of squares within clusters
- k is the number of clusters
- n is the number of samples

**Why use this method:**
- It provides an objective measure of clustering quality
- It balances between having too few large clusters and too many small clusters
- The first local peak when going from small to large cluster counts often represents a good balance

The notebook calculates this criterion for 2-20 clusters and identifies the optimal number as the first local peak.

## Sub-task 6: SOM Projection

The notebook implements a dimensionality reduction approach similar to Self-Organizing Maps (SOM) for visualizing high-dimensional data in 2D space.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
While the notebook mentions SOM, it actually uses t-SNE for projection, which is a common technique for visualizing high-dimensional data.

**How t-SNE works:**
1. Compute pairwise similarities between data points in the original high-dimensional space
2. Try to preserve these similarities in a lower-dimensional space (2D in this case)
3. Places similar points close together and dissimilar points far apart

**Why use this method:**
- It excels at preserving local structure of the data
- It reveals clusters and patterns that might be hidden in high-dimensional space
- It's particularly good for visualization purposes

The resulting plot shows the clusters with different colors, helping to visualize their separation and structure.

## Sub-task 7: K-Medoids Clustering

The notebook implements K-Medoids clustering (specifically the PAM algorithm - Partitioning Around Medoids).

### K-Medoids Algorithm
**How it works:**
1. Select k data points as initial medoids (cluster representatives)
2. Assign each data point to the closest medoid
3. For each cluster, find the point that minimizes the sum of distances to other points in the cluster
4. If these points differ from the current medoids, replace the medoids and repeat from step 2
5. Continue until medoids no longer change

**Key differences from K-Means:**
- Medoids are actual data points (not calculated centroids)
- Can use any distance metric (not just Euclidean)
- More robust to outliers

**Why use K-Medoids:**
- It identifies representative data points (medoids) for each cluster, which are actual examples from the dataset
- It's less sensitive to outliers than K-means
- It allows the use of any distance metric (Manhattan distance in this case)
- It's particularly useful when you want to identify "prototype" examples that best represent each cluster

The notebook identifies the most typical representative (by name) in each cluster, which helps in understanding the characteristics of each cluster.

## Sub-task 8: Implementation as Class

The notebook encapsulates all the previous clustering steps into a `BaseballClustering` class for reusability. This demonstrates good software engineering practices:

- **Encapsulation**: The clustering logic is contained within a class
- **Reusability**: The class can be applied to different datasets
- **Modularity**: Different methods handle distinct parts of the pipeline

The class includes methods for:
- Data preprocessing (normalization and encoding)
- Finding the optimal number of clusters
- Hierarchical clustering and dendrogram creation
- Data projection and visualization
- K-medoids clustering
- Running the complete clustering pipeline

This approach makes the code more maintainable and allows for easier experimentation with different parameters or datasets.

## Sub-task 9: Distribution Transformation

The notebook applies log transformations to make feature distributions more symmetric:

**Why log transformation:**
- Reduces the effect of outliers
- Makes right-skewed distributions more symmetric
- Can improve the performance of distance-based clustering algorithms
- The log(1+x) formula preserves zero values

After transformation, the clustering pipeline is run again to compare results. This demonstrates how preprocessing can affect clustering outcomes.

## Sub-task 10: VarClus Feature Selection

The notebook implements the VarClus method to select the most significant variables:

### VarClus (Variable Clustering)
**How it works:**
1. Compute correlation matrix between variables
2. Convert correlations to distances (1-|correlation|)
3. Apply hierarchical clustering to the variables (not the observations)
4. Cut the dendrogram to get variable clusters
5. Select the most representative variable from each cluster

**Why use VarClus:**
- Reduces dimensionality while preserving most of the variation in the data
- Handles multicollinearity by grouping correlated variables
- Selects features that represent different aspects of the data
- Results in more interpretable models

The notebook selects the 5 most significant variables and compares clustering results on this reduced feature set.

## Sub-task 11: Anomaly Detection with One-Class SVM

This task focuses on unsupervised anomaly detection using the MNIST dataset, specifically identifying digit 6 as anomalies among digit 0 samples.

### One-Class SVM
**How it works:**
1. Learns a boundary around the "normal" data (digit 0)
2. Points outside this boundary are classified as anomalies (digit 6)
3. Uses a kernel function (RBF in this case) to transform the data to a space where the boundary can be defined

**Key parameters:**
- **nu**: Controls the upper bound on the fraction of training errors and the lower bound on the fraction of support vectors (between 0 and 1)
- **gamma**: Defines the influence of each training example (how far it reaches)
- **kernel**: Determines the type of decision boundary (RBF for flexible non-linear boundaries)

**Why use One-Class SVM:**
- Works in an unsupervised setting (doesn't use labels during training)
- Can learn complex boundaries around normal data
- Well-suited for anomaly detection tasks
- Can be tuned through parameters like nu and gamma

The notebook also implements:
- **PCA for dimensionality reduction**: Reduces the 784-dimensional digit images to 50 dimensions
- **Feature engineering**: Creates edge features to improve discrimination
- **Grid search**: To find optimal parameters

## Sub-task 12: ROC Curve and Example Analysis

The notebook evaluates the anomaly detection model using:

### ROC Curve
**What it shows:**
- Plots true positive rate against false positive rate at various thresholds
- Area Under Curve (AUC) quantifies overall performance (higher is better)
- Equal Error Rate (ERR) is where false positive rate equals false negative rate

**Why use ROC:**
- Threshold-independent performance evaluation
- Visualizes the trade-off between detecting anomalies and false alarms
- AUC provides a single metric for model comparison

The notebook also identifies and visualizes:
1. Most typical digit 0 (true negative with minimal abnormality)
2. Most abnormal digit 6 (true positive with maximal abnormality)
3. Most atypical digit 0 (false positive with maximal abnormality)
4. Most non-anomalous digit 6 (false negative with minimal abnormality)

This analysis helps understand what patterns the model is learning and where it might be failing.

## Overall Conclusion

This notebook provides a comprehensive exploration of clustering and anomaly detection techniques, demonstrating:

1. **Data preprocessing**: Handling missing values, normalization, and encoding
2. **Clustering algorithms**: Hierarchical clustering and K-medoids
3. **Cluster validation**: Pseudo-F criterion for determining optimal cluster count
4. **Dimensionality reduction**: t-SNE for visualization
5. **Feature engineering**: Log transformations and VarClus for feature selection
6. **Anomaly detection**: One-class SVM for identifying outliers
7. **Model evaluation**: ROC curves and example analysis

Each technique is applied with a clear understanding of when and why it should be used, demonstrating good practical knowledge of unsupervised learning methods.