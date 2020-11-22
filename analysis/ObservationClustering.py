import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
columns = ["patient_id", "Body Height", "Body Weight", "Body Mass Index", "Body temperature", 
          "Respiratory rate", "Systolic Blood Pressure", "Triglycerides", "Hemoglobin [Mass/volume] in Blood"]
df = pd.DataFrame(columns = columns)
df["patient_id"] = patient_ids

print("Filling NaN values")
patients = []
descriptions = []
values = []
for index, row in data[["PATIENT", "DESCRIPTION", "VALUE"]].iterrows():
    if row["DESCRIPTION"] in columns:
        for column in columns:
            if row["DESCRIPTION"] == column:
                patients.append(row["PATIENT"])
                descriptions.append(column)
                values.append(row["VALUE"])

for index, row in df.iterrows():
    for pat, desc, val in zip(patients, descriptions, values):
        if pat == row["patient_id"]:
            df.loc[df.patient_id == pat, desc] = val

print("Normalizing data")
del df['patient_id']
imp_mean = SimpleImputer(missing_values = np.nan, strategy='mean')
df = imp_mean.fit_transform(df)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
print(pd.DataFrame(df_scaled).head());

print("Performing Sample Clustering")
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
kmeans.fit(df_scaled)
SSE = []
for cluster in range(1, 20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init = "k-means++")
    kmeans.fit(df_scaled)
    SSE.append(kmeans.inertia_)

# Inertia of a clustering job is the sum of the distances of each point in each cluster to their centroid.
# It indicates how "tight" the clusters are, so the smaller this value the better.
# To find the optimum number of clusters, we can graph the inertia as a function of the number of clusters.
# We don't want too many clusters, because this will slow the model down on large datasets.
frame = pd.DataFrame({"Cluster": range(1, 20), "SSE": SSE})
plt.figure(figsize = (12, 6))
plt.plot(frame["Cluster"], frame["SSE"], marker = 'o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()
# Since 15 is not a big number and curve seems to flatten out at that point, that's why I chose 15 as number of clusters

kmeans = KMeans(n_jobs = -1, n_clusters = 15, init = 'k-means++')
kmeans.fit(df_scaled)
pred = kmeans.predict(df_scaled)
print(pred)

frame = pd.DataFrame(df_scaled)
frame["cluster"] = pred
print(frame["cluster"].value_counts())
