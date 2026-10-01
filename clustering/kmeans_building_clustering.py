import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load building data
df = pd.read_csv("../data/sample_buildings.csv")

# Features used for clustering
features = [
    "energy_consumption",
    "occupancy",
    "hvac_runtime",
    "building_area"
]

X = df[features]

# Standardize features because K-Means is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

# Assign each building to a cluster
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nBuilding Cluster Assignments:")
print(df[["building_id", "cluster"]])

# Analyze characteristics of each cluster
cluster_summary = df.groupby("cluster")[features].mean()

print("\nCluster Summary:")
print(cluster_summary)
