import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# 1) Loaded data
df = pd.read_csv("top_insta_influencers_data.csv")

# 2) Data Cleaning / Preprocessing
# Converting strings like "3.3k", "475.8m", "29.0b", "1.62%" into numbers
def convert_to_number(x):
    if isinstance(x, str):
        s = x.lower().replace(",", "").strip()
        try:
            if s.endswith("k"):
                return float(s[:-1]) * 1000
            if s.endswith("m"):
                return float(s[:-1]) * 1000000
            if s.endswith("b"):
                return float(s[:-1]) * 1000000000
            if s.endswith("%"):
                return float(s[:-1])

            return float(s)
        except:
            return np.nan
    return x

# Columns that need numeric conversion
to_numeric = ["Posts","Followers","Average Likes","Engagement Rate","New post average like","Total Likes"]

for col in to_numeric:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_number)

# Filling nan values with mode values
if "Country" in df.columns:
    df["Country"] = df["Country"].fillna(df['Country'].mode()[0])

# Dropping rows that are missing the most essential numeric fields
df = df.dropna(subset=["Followers", "Average Likes", "Engagement Rate"]).reset_index(drop=True)

# Mathematical Calculations
df["Likes_per_1k_followers"] = (df["Average Likes"] / df["Followers"]) * 1000
df["Posts_per_1k_followers"] = (df["Posts"] / df["Followers"]) * 1000


# 3) Visualizations / Graphs
# 1. Top 20 by Followers (line chart)
top20 = df.nlargest(20, "Followers")
plt.figure(figsize=(10,5))
plt.plot(top20["Name"], top20["Followers"])
plt.xticks(rotation = 45, ha = "right")
plt.xlabel("Name of the Influencer", fontweight = 'bold')
plt.ylabel("Number of Followers [in Millions]", fontweight = 'bold')
plt.title("Top 20 Influencers by Followers", fontweight = 'bold')
plt.tight_layout()
plt.show()

# 2. Followers vs Average Likes (scatter plot)
plt.figure(figsize=(8,5))
plt.scatter(df["Followers"], df["Likes_per_1k_followers"], alpha = 0.5 ,color= "red")
plt.xlabel("Number of Followers (in 100 Millions)", fontweight = 'bold')
plt.ylabel("Average Likes Per Post", fontweight = 'bold')
plt.title("Followers vs Average Likes per Post", fontweight = 'bold')
plt.tight_layout()
plt.show()

# 3. Engagement Rate distribution (histogram)
plt.figure(figsize=(8,5))
plt.hist(df["Engagement Rate"], bins = 20 ,color = "pink")
plt.xlabel("Engagement Rate (in %)", fontweight = 'bold')
plt.ylabel("Number of Influencers", fontweight = 'bold')
plt.title("Influencers vs their Engagement Rate Distribution", fontweight = 'bold')
plt.tight_layout()
plt.show()

# 4. Country-wise influencer count (Top 20)  (bar graph)
country_counts = df["Country"].value_counts().head(20)
plt.figure(figsize=(10,5))
plt.bar(country_counts.index, country_counts.values, color = "green")
plt.xticks(rotation = 45, ha = "right")
plt.xlabel("Country's Name", fontweight = 'bold')
plt.ylabel("Number of Influencers", fontweight = 'bold')
plt.title("Top Countries by Influencer Count", fontweight = 'bold')
plt.tight_layout()
plt.show()

# 5. Linear Regression (Supervised): Predicting Average Likes from Followers
x = df[["Followers"]]
y = df["Average Likes"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Prediction on test set
y_pred = linreg.predict(X_test)

# Plotting test points & regression line
plt.figure(figsize=(8,5))
plt.scatter(X_test["Followers"], y_test, alpha = 0.6, label = "Actual")

sorted_idx = np.argsort(X_test["Followers"].values.ravel())
plt.plot(X_test["Followers"].values.ravel()[sorted_idx],y_pred[sorted_idx],linewidth = 2, label="Predicted")

plt.xlabel("Number of Followers (in 100 Millions)", fontweight = 'bold')
plt.ylabel("Average Likes (in Millions)", fontweight = 'bold')
plt.title("Linear Regression: Followers -> Average Likes", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.show()

# Printing a simple score
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
print("\nLinear Regression R^2 on test set:", linreg.score(X_test, y_test))
print("--------------------------------------------------------------------------------------------------------------------------------------------------------")


# 6. KMeans (Unsupervised): Cluster on Followers & Engagement Rate
x_cluster = df[["Followers", "Engagement Rate"]]

kmeans = KMeans(n_clusters = 5, random_state = 100)
df["Cluster"] =  kmeans.fit_predict(x_cluster)

cluster_labels = {
    0: "Nano Influencers" ,
    1: "Micro Influencers" ,
    2: "Mid-tier Influencers",
    3: "Macro Influencers" ,
    4: "Mega Influencers"
}

df["Cluster_Name"] = df["Cluster"].map(cluster_labels)

plt.figure(figsize=(8,5))
for cluster_name , cluster_data in df.groupby("Cluster_Name"):
    plt.scatter(cluster_data["Followers"], cluster_data["Engagement Rate"], label = cluster_name, alpha=0.7)
plt.xlabel("Number of Followers (in 100 Millions)", fontweight = 'bold')
plt.ylabel("Engagement Rate (in %)", fontweight = 'bold')
plt.title("KMeans Clustering of Influencers", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.show()

# Printing a small preview 
print("\n\n--------------------------------------------------------Cleaned data preview-------------------------------------------------------------------------\n")
print(df.head(10).to_string(index = False))

