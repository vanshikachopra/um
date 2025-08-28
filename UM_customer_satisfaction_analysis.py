import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

# 1. Loading the Data
train = pd.read_csv("customer_train_dataset.csv")
test = pd.read_csv("customer_test_dataset.csv")

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

# 2. Data Cleaning & Preprocessing
train.columns = train.columns.str.lower().str.replace(" ", "_").str.replace("%", "pct")
test.columns = test.columns.str.lower().str.replace(" ", "_").str.replace("%", "pct")

for col in train.select_dtypes(include=[np.number]).columns:
    train[col] = train[col].fillna(train[col].median())
    if col in test.columns:
        test[col] = test[col].fillna(train[col].median())


# 3. Exploratory Data Analysis {EDA}

# Graph 1: Rating Distribution                                    (line chart)
rating_counts = train["rating"].value_counts().sort_index()
plt.plot(rating_counts.index, rating_counts.values, color = "skyblue", marker = "o")
plt.title("Rating Distribution" ,fontweight = "bold")
plt.xlabel("Rating Scale" ,fontweight = "bold")
plt.ylabel("Number of Customers",fontweight = "bold")
plt.tight_layout()
plt.show()



# Graph 2: Average Rating by Platform                             (bar graph)
if "platform" in train.columns:
    avg_rating_by_platform = train.groupby("platform")["rating"].mean()
    avg_rating_by_platform.plot(kind="bar", color="teal")
    plt.title("Average Rating by Platform", fontweight="bold")
    plt.ylabel("Average Rating", fontweight="bold")
    plt.xlabel("Name of the Platform", fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# Graph 3: Price vs Rating                                              (scatter graph)
if "actprice1" in train.columns:
    plt.scatter(train["actprice1"], train["rating"], alpha = 0.5, color = "purple" , marker = "x")
    plt.title("Product's Price vs Rating" ,fontweight = "bold")
    plt.xlabel("Price of the Products" ,fontweight = "bold")
    plt.ylabel("Customer's Rating" ,fontweight = "bold")
    plt.tight_layout()
    plt.show()



# Graph 4: Review Count Distribution                                               (scatter plot)
if "noreviews1" in train.columns and "rating" in train.columns:
    plt.scatter(train["noreviews1"], train["rating"], alpha=0.5, color="red")
    plt.title("Review Count vs Rating", fontweight="bold")
    plt.xlabel("Number of Reviews", fontweight="bold")
    plt.ylabel("Customer's Rating Scale", fontweight="bold")
    plt.tight_layout()
    plt.show()


# Graph 5: Star Distribution                                                        (pie chart)
stars = [c for c in train.columns if "star_" in c]
if stars:
    avg_stars = train[stars].mean()
    avg_stars.index = avg_stars.index.str.replace("star_", "").str.replace("f", "") + " Star"

    explode = [0.05] * len(avg_stars)  
    plt.pie(avg_stars, labels = avg_stars.index , autopct = "%1.1f%%", colors = ["pink","magenta","teal","violet","skyblue"], shadow = True, explode = explode, textprops = {"fontweight": "bold"})
    plt.title("Customer Rating Distribution (Average Proportion)", fontweight = "bold")
    plt.tight_layout()
    plt.show()



# 4. Preparing Data for the Model
# Targets
y_reg = train["rating"]                        # for regression
y_clf = (train["rating"] >= 4).astype(int)     # for classification

# Dropping non-useful columns
drop_cols = ["id", "title", "rating"]
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns], errors="ignore")

# Converting categorical data to numbers
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Aligning train/test columns
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Spliting training data
X_train, X_valid, y_reg_train, y_reg_valid = train_test_split(X, y_reg, test_size=0.2, random_state=100)
_, _, y_clf_train, y_clf_valid = train_test_split(X, y_clf, test_size=0.2, random_state=100)



# 5. Regression Model (predicts exact rating)
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_valid)

mse = mean_squared_error(y_reg_valid, y_reg_pred)
rmse = np.sqrt(mse)
print("\nRegression RMSE:", rmse)



# 6. Classification Model
clf_model = LogisticRegression(solver="liblinear", max_iter = 1000, C = 1.0, random_state = 100)
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_valid)

acc = accuracy_score(y_clf_valid, y_clf_pred)
f1 = f1_score(y_clf_valid, y_clf_pred)
print("\nClassification Accuracy:", acc)
print("\nClassification F1 Score:", f1)



