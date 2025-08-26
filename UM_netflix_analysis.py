import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def clean_column_name(s):
    return re.sub(r'[\W_]+', '_', s).strip('_').lower()

def parse_duration(val):
    if not isinstance(val, str):
        return np.nan, np.nan
    s = val.lower().strip()
    if "min" in s:
        return int(s.split()[0]), np.nan
    if "season" in s:
        return np.nan, int(s.split()[0])
    return np.nan, np.nan


# 1) Loading the dataset
file = "netflix_dataset.csv"   
df = pd.read_csv(file)
print("Original data shape:", df.shape)

# 2) Data Cleaning
df.columns = [clean_column_name(c) for c in df.columns]  
df = df.drop_duplicates()                       
df = df.fillna("Unknown")                       

# 3) Feature Engineering
# Release year
if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# Duration (splitting into minutes(movies) vs seasons(shows))
minutes, seasons = [], []
if "duration" in df.columns:
    for v in df["duration"]:
        m, s = parse_duration(v)
        minutes.append(m)
        seasons.append(s)
df["duration_minutes"] = minutes
df["seasons_count"] = seasons

# Title length
df["title_length"] = df["title"].astype(str).apply(len)


# 5) Analysis & Visualizations
# 1. Count of Movies vs TV Shows                                                        (pie chart)
df["type"].value_counts().plot(kind = "pie", autopct = "%1.1f%%", startangle = 90, shadow = True, explode=[0.05, 0], colors = ["purple", "pink"])
plt.title("Number of Movies vs TV Shows in the dataset", fontweight = "bold")
plt.ylabel("")
plt.tight_layout()
plt.show()



# 2. Titles by Release Year                                                             (line chart)
df["release_year"].dropna().astype(int).value_counts().sort_index().plot(kind = "line", color = "red")
plt.title("Number of Movies/TV Shows released in each year", fontweight = "bold")
plt.xlabel("Release Year", fontweight = "bold")
plt.ylabel("Number of Movies/TV Shows", fontweight = "bold")
plt.tight_layout()
plt.show()



# 3. Ratings Distribution (K-Means Clustering)                                          (scatter plot)
rating_map = {
    "TV-Y": 6,
    "TV-Y7": 7,
    "TV-G": 7,
    "G": 7,
    "PG": 10,
    "TV-PG": 10,
    "PG-13": 13,
    "TV-14": 14,
    "R": 17,
    "NC-17": 18,
    "TV-MA": 18
    }

# Mapping according to netflix rating criteria
df["rating_age"] = df["rating"].map(rating_map)
df["rating_age"] = df["rating_age"].fillna(df["rating_age"].median())

# Features for better KMeans clustering
X_scaled = df[["release_year", "rating_age"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaled)
n_clusters = 4
kmeans = KMeans(n_clusters, random_state = 100)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Dictionary mapping cluster number to custom label
cluster_labels = {
    0: "U/A 7+: Kids Content",
    1: "U/A 13+: Family/Teens",
    2: "U/A 16+: Young Adults",
    3: "A: Adults"
}

# Scatter plot with colormap
scatter = plt.scatter(df["release_year"], df["rating_age"],c=df["cluster"], cmap="tab10", alpha=0.6)

# Creating legend handles
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)

# Replacing auto labels with dictionary mapping
new_labels = [cluster_labels[int(i)] for i in range(n_clusters)]
plt.legend(handles, new_labels, title="Clusters")
plt.title("Clusters by Release Year & Audience Age", fontweight="bold")
plt.xlabel("Release Year", fontweight="bold")
plt.ylabel("Age-wise Rating", fontweight="bold")
plt.tight_layout()
plt.show()



# 4. Movie Durations                                                                (histogram)
movies = df[df["type"] == "Movie"]
plt.hist(movies["duration_minutes"], bins=30, color = "maroon")
plt.title("Distribution of Movie Durations (in Minutes)", fontweight = "bold")
plt.xlabel("Duration of Movies(in Minutes)", fontweight = "bold")
plt.ylabel("Number of Movies", fontweight = "bold")
plt.tight_layout()
plt.show()



# 5. TV Show Seasons                                                                   (line chart)
shows = df[df["type"] == "TV Show"]
season_counts = shows["seasons_count"].value_counts().sort_index()
plt.plot(season_counts.index, season_counts.values, color = "maroon", marker = "o")
plt.title("Distribution of TV Show Seasons" , fontweight = "bold")
plt.xlabel("Number of Seasons" , fontweight = "bold") 
plt.ylabel("Number of TV Shows" , fontweight = "bold")
plt.tight_layout()
plt.show()



# 6. Top Countries Producing the Netflix Movies/TV Shows in the dataset                 (bar graph)
top_countries = df[df["country"] != "Unknown"]["country"].value_counts().head(15).dropna().plot(kind = "bar" , color = "grey")
plt.title("Top 15 Countries Producing the Netflix Movies/TV Shows in the dataset" , fontweight = "bold")
plt.xlabel("Country's Name" , fontweight = "bold")
plt.xticks(rotation = 60)
plt.ylabel("Number of Movies/TV Shows" , fontweight = "bold")
plt.tight_layout()
plt.show()



# 7. Top Genres                                                                         (pie chart)
top_genres = df["listed_in"].str.split(",").explode().str.strip().value_counts().head(10)
explode = [0.05] * len(top_genres)
top_genres.plot(kind = "pie", autopct = "%1.1f%%", shadow = True, explode = explode, startangle = 0, cmap = "tab10")
plt.title("Distribution of Movies/TV Shows based on their Genre" , fontweight = "bold")
plt.ylabel("")
plt.tight_layout()
plt.show()



# 8. Movies/TV Shows added over Time                                                    (line graph)
# Year-Wise Distribution of Movies/TV Shows added
if "date_added" in df.columns:
    # Converting to datetime
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    # Grouping by year
    titles_per_year = df["date_added"].dt.year.value_counts().sort_index()
    # Plotting the graph
    plt.plot(titles_per_year.index, titles_per_year.values, marker="o", color="blue")
    plt.title("Number of Movies/TV Shows added per Year", fontweight = "bold")
    plt.xlabel("Year", fontweight = "bold")
    plt.ylabel("Number of Titles", fontweight = "bold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Month-Wise Distribution of Movies/TV Shows added
if "date_added" in df.columns:
    # Converting to datetime
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    # Grouping by month 
    titles_per_month = df["date_added"].dt.month.value_counts().sort_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Plotting the graph
    plt.plot(month_names, titles_per_month.values, marker = "o", color = "purple")
    plt.title("Number of Movies/TV Shows added per Month (All Years Combined)", fontweight = "bold")
    plt.xlabel("Name of the Month", fontweight = "bold")
    plt.ylabel("Number of Titles", fontweight = "bold")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




