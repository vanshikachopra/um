import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Loading and combining datasets
df1 = pd.read_csv("coffee sales1.csv")
df2 = pd.read_csv("coffee sales2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Data Cleaning
df.drop(columns = ['card'], inplace = True)
df.drop_duplicates(inplace = True)

# Data Preprocessing
df['is_card'] = df['cash_type'].apply(lambda x: 1 if x == 'card' else 0)
df['datetime'] = pd.to_datetime(df['datetime'], format = "mixed")
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.day_name()
df['month'] = df['datetime'].dt.month_name()

# Encoding categorical variables
df_encoded = pd.get_dummies(df[['cash_type', 'coffee_name', 'hour']], drop_first=True)

# Preparing features and target
X = df_encoded
y = df['money']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting and evaluating
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Linear Regression Model - Mean Squared Error: {mse:.2f}")


# 1. Coffee popularity
plt.figure(figsize=(10,6))
sns.countplot(data = df, y ='coffee_name', order = df['coffee_name'].value_counts().index)
plt.title("Most Popular/Preffered Coffees among Customers" , fontweight = 'bold')
plt.xlabel("Number of Cups sold" , fontweight = 'bold')
plt.ylabel("Coffee Type" , fontweight = 'bold')
plt.tight_layout()
plt.show()

# 2. Payment method distribution
plt.figure(figsize=(6,6))
df['cash_type'].value_counts().plot.pie(autopct = '%1.1f%%', startangle = 90)
plt.title("Payment Method Distribution[Card/Cash]" , fontweight = 'bold')
plt.ylabel("")
plt.tight_layout()
plt.show()

# 3. Hourly sales pattern
plt.figure(figsize=(10,6))
sns.histplot(data = df, x ='hour', bins = 24, kde = True)
plt.title("Customer Purchase Activity by Time of the Day" , fontweight = 'bold')
plt.xlabel("Hour (24- hour format)" , fontweight = 'bold')
plt.ylabel("Numbers of Transactions" , fontweight = 'bold')
plt.tight_layout()
plt.show()

# 4. Heatmap: Sales by Weekday and Hour
heatmap_data = df.groupby(['weekday', 'hour']).size().unstack().fillna(0)
heatmap_data = heatmap_data.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap='YlOrBr')
plt.title("Sales Heatmap: Time of the Day vs Day of the Week" , fontweight = 'bold')
plt.xlabel("Hour of the Day" , fontweight = 'bold')
plt.ylabel("Days of the Week" , fontweight = 'bold')
plt.tight_layout()
plt.show()


