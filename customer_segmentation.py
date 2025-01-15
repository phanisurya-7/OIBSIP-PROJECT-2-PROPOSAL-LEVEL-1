import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('ifood_df.csv')

# Step 2: Explore the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 3: Handle missing values (if any)
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Handle duplicates (if any)
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Drop duplicate rows if found
data = data.drop_duplicates()

# Step 5: Create a 'Total Spend' column by summing the spending columns
data['Total Spend'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                             'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Step 6: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 7: Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # You can experiment with the number of clusters
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 8: Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Total Spend'], hue=data['Cluster'], palette='Set1', s=100)
plt.title('Customer Segments based on Age and Total Spend')
plt.xlabel('Age')
plt.ylabel('Total Spend')
plt.show()

# Step 9: Descriptive Statistics for Each Cluster
print("\nCluster Summary:")
print(data.groupby('Cluster').agg({'Age': ['mean', 'std'], 'Total Spend': ['mean', 'std']}))

# Step 10: Provide insights and recommendations based on the clusters
for cluster in range(5):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nInsights for Cluster {cluster}:")
    print(cluster_data.describe())
    print("----")
