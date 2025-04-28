# Detailed Explanation of Algorithms and Methods in Association Rules and Hidden Structures Notebook

## Task 1: Finding Association Rules and Revealing Hidden Structures in Data

This notebook explores market basket analysis and pattern discovery in transaction data using various data mining and machine learning techniques. Let's examine each subtask in detail:

### Subtask 1: Load the Data, Determine Unique Counts

- Initial step in any data analysis project to understand the dataset's dimensions
- Helps determine the cardinality of categorical variables before proceeding with more complex analyses
- Provides insights into potential computational complexity (e.g., large number of products could impact performance)


### Subtask 2: Find Frequent Episodes using FPTree Algorithm

**FP-Growth (Frequent Pattern Growth)**

- FP-Growth is a pattern mining algorithm that discovers frequent itemsets in transaction data
- It builds a compressed FP-tree data structure that stores frequent patterns efficiently
- Unlike Apriori, it scans the database only twice, making it more efficient

**When and Why Used:**
- Used when discovering frequent patterns or co-occurring items in transaction data
- Particularly valuable in retail for market basket analysis to identify products frequently purchased together
- More efficient than Apriori algorithm for large datasets, especially those with many transactions or items
- Used here to identify combinations of products that frequently appear together in customer purchases

**Implementation Details:**
- Data is first transformed into transaction format where each row represents a customer's basket
- `TransactionEncoder` converts these transactions into a binary matrix (one-hot encoded)
- `min_support=0.05` means only itemsets appearing in at least 5% of transactions are considered
- `max_len=4` constrains the algorithm to find only itemsets with up to 4 items
- `use_colnames=True` preserves product names in the results instead of numerical indices

### Subtask 3: Find the Largest Frequent Episode containing "peppers"

**Filtering and Finding Maximum**

- Filters the frequent itemsets found in Subtask 2 to include only those containing "peppers"
- Identifies the itemset with the maximum number of elements among those filtered results

**When and Why Used:**
- Used when focusing on specific items of interest in pattern mining
- Helpful for product-specific analysis to understand what other products commonly appear with a target product
- Can inform marketing strategies like product bundling or targeted promotions

### Subtask 4: Construct Association Rules with Reliability Threshold of 30%

**Association Rule Mining**

- Generates association rules from the frequent itemsets discovered previously
- Association rules have the form "if A then B" (A → B), where A is the antecedent and B is the consequent
- Rules are filtered by confidence (reliability) threshold and then further filtered to those with "peppers" in the antecedent
- The rule with the maximum lift value is identified

**When and Why Used:**
- Used to discover meaningful relationships between items beyond just co-occurrence
- Provides actionable insights for marketing strategies, store layout, recommendation systems
- Confidence helps determine predictive strength (if A happens, how likely is B?)
- Lift measures how much more likely items occur together versus by random chance
- Particularly valuable when focusing on specific products for targeted marketing

**Implementation Details:**
- `min_threshold=0.3` means only rules with at least 30% confidence are returned
- The rules are evaluated based on three key metrics:
  1. **Support**: Frequency of the entire rule (A and B together) in the dataset
  2. **Confidence**: Conditional probability of B given A: $$P(B|A) = \frac{P(A,B)}{P(A)}$$
  3. **Lift**: Ratio of observed support to expected support if A and B were independent: $$\text{Lift}(A→B) = \frac{P(A,B)}{P(A) \cdot P(B)}$$

### Subtask 5: Construct Directed Graph from Two-Place Rules

**Network Graph Visualization**

**What It Does:**
- Creates a directed graph representation of association rules where:
  - Nodes represent individual products
  - Edges represent association rules between single products (A → B)
  - Node size corresponds to the overall support (popularity) of each product
  - Edge weight represents the confidence (reliability) of the rule

**When and Why Used:**
- Network visualization is used when relationships between entities are important
- Provides an intuitive visual representation of complex relationships in transaction data
- Helps identify central products, clusters of related products, and strong associations
- Particularly useful for identifying influential products in the product ecosystem

**Implementation Details:**
- Rules are filtered to include only those with single item antecedents and consequents (two-place rules)
- Product support values are calculated using `value_counts(normalize=True)` to get relative frequencies
- The graph is visualized using `spring_layout` which positions nodes based on force-directed placement
- Node sizes are scaled by product support, and edge widths are scaled by rule confidence

### Subtask 6: Calculate Centrality Measures and Clustering Coefficient

**Graph Clustering Coefficient Analysis**

- Calculates the clustering coefficient for each node in the association rule graph
- The clustering coefficient measures how interconnected a node's neighbors are with each other

**When and Why Used:**
- Used in network analysis to identify nodes that form tightly connected groups
- High clustering coefficient indicates that a product's associated products also tend to be associated with each other
- Can reveal products that are central to specific market segments or product categories
- Helps identify products that might serve as "bridges" between different product clusters

**Implementation Details:**
- `clustering(G)` from NetworkX calculates the clustering coefficient for all nodes
- For a directed graph, this examines triangles involving the node and its neighbors
- A higher coefficient (closer to 1) indicates a node whose neighbors are well-connected to each other
- The product with the highest coefficient is found using the `max()` function with a key parameter

### Subtask 7: Build Numeric Purchase Matrix

**Pivot Table Transformation**

- Transforms the transaction data from a long format to a wide format
- Creates a matrix where rows represent customers, columns represent products, and values represent purchase counts

**When and Why Used:**
- Provides a compact representation of purchase behavior across all customers and products

**Implementation Details:**
- `pivot_table()` restructures the data from its original form
- `aggfunc="size"` counts occurrences (i.e., number of purchases)

### Subtask 8: NMF Linear Projection

**Non-Negative Matrix Factorization**

- NMF decomposes the purchase matrix into two lower-rank matrices: W and H
- W (returned by `fit_transform()`) represents customers in the reduced space
- H (available as `nmf.components_`) represents how products relate to the components
- Projects high-dimensional purchase data onto a 2D plane for visualization

**When and Why Used:**
- Used for dimensionality reduction when dealing with non-negative data (like purchase counts)
- Allows visualization of complex high-dimensional patterns in a 2D space
- Can reveal latent patterns in purchasing behavior that aren't obvious in the original data
- Unlike PCA, NMF produces components that are more interpretable as they represent additive combinations of features

**Implementation Details:**
- `n_components=2` specifies that the data should be projected onto 2 dimensions
- After projection, data points are labeled based on whether they contain the "peppers" product

### Subtask 9: SOM Nonlinear Projection

**Self-Organizing Map**

- SOM is a type of artificial neural network that produces a discrete low-dimensional representation of the input space
- It creates a topological mapping where similar data points in the high-dimensional space are mapped to nearby locations on a 2D grid
- The algorithm trains a grid of neurons to respond to different input patterns, with neighboring neurons responding to similar patterns

**When and Why Used:**
- Used for nonlinear dimensionality reduction and visualization of high-dimensional data
- Particularly valuable for discovering clusters or patterns that have complex, nonlinear relationships
- Preserves topological properties of the data, meaning similar customers will be mapped to nearby locations
- Can reveal more complex structures than linear methods like PCA or NMF

**Implementation Details:**
- Data is first scaled to [0,1] range using `MinMaxScaler()` as SOM works best with normalized data
- A 10×10 grid of neurons is initialized (creating a 2D map of 100 positions)
- `input_len` is set to the number of products in the dataset
- `sigma=1.0` controls the initial neighborhood radius (how far the influence of each training example reaches)
- `learning_rate=0.5` determines how quickly the model adapts to the training data
- `train_random()` trains the SOM for 1000 iterations using randomly selected samples
- Each customer is then mapped to their "winning" neuron (the neuron whose weights most closely match the customer's purchase pattern)
- The distance map visualization shows the distances between neighboring neurons, with darker areas potentially indicating cluster boundaries

### Subtask 10: Select 6 Independent Variables using Stepwise Selection

**Recursive Feature Elimination (RFE)**

- RFE is a feature selection method that works by recursively removing features
- It fits a model (Logistic Regression in this work), ranks features by importance, and eliminates the least important ones
- The process repeats until the desired number of features remains

**When and Why Used:**
- Used when dealing with high-dimensional data to select the most relevant features
- Helps reduce overfitting, improve model performance, and increase interpretability
- More principled than manual feature selection as it considers feature importance in the context of a predictive model
- In marketing analytics, helps identify which products are most predictive of purchasing a target product

**Implementation Details:**
- Creates a binary target variable indicating whether "peppers" was purchased
- Uses Logistic Regression as the estimator model, which is appropriate for binary classification tasks
- `n_features_to_select=6` specifies that only 6 features should be retained
- RFE works by:
  1. Training the model with all features
  2. Ranking features by importance
  3. Eliminating the least important feature(s)
  4. Repeating until only the specified number of features remains
- The selected features are those most predictive of whether a customer purchases peppers

## Summary of Methods and Their Applications

Each algorithm in this notebook serves a specific purpose in the data analysis pipeline:

1. **Data Exploration (Subtask 1)**: Understanding dataset dimensions and characteristics
2. **FP-Growth (Subtask 2)**: Efficient discovery of frequent itemsets in transaction data
3. **Association Rule Mining (Subtasks 3-4)**: Revealing meaningful relationships between products
4. **Network Analysis (Subtasks 5-6)**: Visualizing and analyzing the structure of product relationships
5. **Pivot Table Transformation (Subtask 7)**: Preparing data for machine learning algorithms
6. **NMF (Subtask 8)**: Linear dimensionality reduction preserving non-negativity
7. **SOM (Subtask 9)**: Nonlinear projection preserving topological properties
8. **RFE (Subtask 10)**: Principled feature selection based on predictive importance

Together, these methods provide a comprehensive approach to discovering and interpreting patterns in transaction data, progressing from basic pattern discovery to more sophisticated machine