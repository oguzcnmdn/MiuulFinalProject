###################################################################################################
############################   MODEL CREATION   ###################################################
###################################################################################################

import pandas as pd

df = pd.read_csv('final_traffic_data_C.csv')

df.drop(columns="Unnamed: 0", axis= 1, inplace=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "0"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th= 31, car_th=20)

df.head()
df.isnull().sum()
df.info()
df.describe().T

for col in df.columns:
    if df[col].isnull().any():
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

### SCALING ###

from sklearn.preprocessing import RobustScaler

df_ng = df.drop(columns='GEOHASH', axis=1)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_ng)


### K-MEANS ###

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(scaled_data)
elbow.show()

elbow.elbow_value_

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(scaled_data)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
scaled_data[0:5]

clusters_kmeans = kmeans.labels_

df["cluster"] = clusters_kmeans

df.head()

df["cluster"] = df["cluster"] + 1

df.to_csv("clustered_final.csv")

df_cluster = df[df['cluster'] == 5]

import folium

m = folium.Map(location=[df_cluster['LATITUDE'].mean(), df_cluster['LONGITUDE'].mean()], zoom_start=10)

for index, row in df_cluster.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('cluster_5.html')


### PCA ###

pca = PCA(n_components = 8)
pca.fit(df)

x_pca = pca.transform(df)

print("Variance Ratio", pca.explained_variance_ratio_)
print("Sum:", sum(pca.explained_variance_ratio_))

# TODO: Model to be created.

