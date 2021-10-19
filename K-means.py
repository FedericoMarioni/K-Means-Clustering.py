# K-means . Unsupervised learning - clustering
#  Means Clustering Algorithm | K Means Example in Python | Machine Learning Algorithms | Edureka
# K : parameter number of clusters
# customer segmentation

# Initialisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data set

df = pd.read_csv("Mall_Customers.csv")

# print(df.head())

df.rename(columns={'Genre': 'Gender'}, inplace=True)

print(df.shape)  # (n, Xj) Number of samples and variables

print(df.describe())  # Descriptive statistics for each numerical variable

print(df.dtypes)  # Check the data types for each variable

print(df.isnull().sum())  # Check for null values

df.drop(["CustomerID"], axis=1, inplace=True)  # Dropping CustomerID column

# Analysis and visualization of the data

plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.distplot(df[x], bins=20)
    plt.title('Distplot of {}'.format(x))

# plt.show()  # Distribution for each numerical variable

plt.figure(figsize=(15, 5))
sns.countplot(y='Gender', data=df)

# plt.show()  # Distribution for categorical variable

# Violin Plot of Age, Annual Income, Spending Score depending on sex = "F", "M"


plt.figure(1, figsize=(15, 7))
n = 0
for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    sns.set(style="whitegrid")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.violinplot(x=cols, y='Gender', data=df)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Violin plot')

# plt.show()  # Conditional distribution

# Divide Age variable into different intervals

age_18_25 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
age_26_35 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
age_36_45 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
age_46_55 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
age_55above = df.Age[(df.Age >= 56)]

agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
agey = [len(age_18_25.values), len(age_26_35.values), len(age_36_45.values), len(age_46_55.values),
        len(age_55above.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=agex, y=agey, palette="mako")
plt.title("Number of customers per age intervals")
plt.xlabel("Age")
plt.ylabel("Number of customers")
# plt.show()  # Bar plot for distribution of Age in intervals

# Let´s try to understand the relation between Annual Income and Spending Score

sns.relplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df)
# plt.show() # There does not seem to be a linear relation
# But there is some relation in Annual Income 40-60k and Spending Score 40-60

# Divide spending score into different intervals

ss_1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss_21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss_41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss_61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss_81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss_1_20.values), len(ss_21_40.values), len(ss_41_60.values), len(ss_61_80.values), len(ss_81_100.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=ssx, y=ssy, palette="rocket")
plt.title("Number of customers per spending scores intervals")
plt.xlabel("Score")
plt.ylabel("Number of customer having the score")

# plt.show(). Bar plot for distribution of Spending Scores in intervals

# Divide annual income into different intervals

ai_0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai_31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai_61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai_91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai_121_150= df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

aix = ["$ 0 - 30.000", "$30.001 - 60.000", "60.001 - 90.000", "$90.001 - 120.000", "$120.001 - 150.000"]
aiy = [len(ai_0_30.values), len(ai_31_60.values), len(ai_61_90.values), len(ai_91_120.values),
       len(ai_121_150.values)]

plt.figure(figsize=(15, 6))
sns.barplot(x=aix, y=aiy, palette="Spectral")
plt.title("Number of customers per annual income intervals ")
plt.xlabel("Income")
plt.ylabel("Number of customer")

# plt.show() Bar plot for distribution of Annual Income in intervals

# Running the k-means clustering algorithm

# Find out the optimal k parameter (number of clusters)

# Model 1. Cluster the data using 2 variables: Age and Spending Score

X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values

from sklearn.cluster import KMeans

# Optimal k

wcss=[]

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1,11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("value")
plt.ylabel("WCSS")
# plt.show() # After 4 when we increase k WCSS is almost constant

kmeans11 = KMeans(n_clusters=4)  # Initialize the class object

label = kmeans11.fit_predict(X1)   # Predict the labels of clusters

print(label)   # Verification of k = 4

print(kmeans11.cluster_centers_)  # 4 centroids with their respective age and annual income

plt.scatter(X1[:, 0], X1[:, 1], c=kmeans11.labels_, cmap='rainbow')
plt.scatter(kmeans11.cluster_centers_[:, 0], kmeans11.cluster_centers_[:, 1], color='black')
plt.title("Clusters of Customers")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")

# plt.show()


# Model 2. Cluster the data using two variables: Annual Income and Spending Score

X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values

WCSS2 = []

for k in range(1,11):
    kmeans2 = KMeans(n_clusters=k, init="k-means++")
    kmeans2.fit(X2)
    WCSS2.append(kmeans2.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1,11), WCSS2, linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS2")

# plt.show() Choose K = 5

kmeans21 = KMeans(n_clusters=5)

label = kmeans21.fit_predict(X2)

print(label)   # k = 5

print(kmeans21.cluster_centers_)

plt.scatter(X2[:, 0], X2[:, 1], c=kmeans21.labels_, cmap='rainbow')
plt.scatter(kmeans21.cluster_centers_[:, 0], kmeans21.cluster_centers_[:, 1], color='black')
plt.title("clusters of customers")
plt.xlabel("Annual Income ()")
plt.ylabel("Spending Score (1-100)")

# plt.show()




