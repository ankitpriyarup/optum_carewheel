import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


print("Reading data")
data = pd.read_csv("../data/observations.csv")
patient_ids = set()
for patient_id in data["PATIENT"]:
    patient_ids.add(patient_id)
patient_ids = list(patient_ids)
patient_ids.sort()
height_and_weight = pd.DataFrame(columns = ["patient_id", "Body Height", "Body Weight"])
height_and_weight["patient_id"] = patient_ids
height_and_weight.head()

print("Filling NaN values")
patients_height = []
patients_weight = []
heights = []
weights = []
for index, row in data[["PATIENT", "DESCRIPTION", "VALUE"]].iterrows():
    if row["DESCRIPTION"] == "Body Height":
        patients_height.append(row["PATIENT"])
        heights.append(row["VALUE"])
    elif row["DESCRIPTION"] == "Body Weight":
        patients_weight.append(row["PATIENT"])
        weights.append(row["VALUE"])

for index, row in height_and_weight.iterrows():
    i = 0 
    j = 0
    for patient in patients_height:
        if row["patient_id"] == patient:
            height_and_weight.loc[height_and_weight.patient_id == patient, "Body Height"] = heights[i]
        i += 1
    for patient in patients_weight:
        if row["patient_id"] == patient:
            height_and_weight.loc[height_and_weight.patient_id == patient, "Body Weight"] = weights[j]
        j += 1

print("Normalizing data")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(height_and_weight[["Body Height", "Body Weight"]])
pd.DataFrame(data_scaled).head()

print("Performing Sample Clustering")
K = 2
kmeans = KMeans(n_clusters = K, init = 'k-means++')
kmeans.fit(data_scaled)
SSE = []
for cluster in range(1, 20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init="k-means++")
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
    
frame = pd.DataFrame({"Cluster": range(1, 20), "SSE": SSE})
plt.figure(figsize = (12, 6))
plt.plot(frame["Cluster"], frame["SSE"], marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init = 'k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame["cluster"] = pred
frame["cluster"].value_counts()

color = ["blue", "green", "navy", "cyan", "purple"]
for k in range(5):
    data = frame[frame["cluster"]  == k + 1]
    plt.scatter(data[0], data[1], c=color[k])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()
