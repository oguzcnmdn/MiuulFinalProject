###################################################################################################
############################   MODEL CREATION   ###################################################
###################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

data = pd.read_csv('final_traffic_data_C.csv')

df = data.copy()

df.drop(columns=['Unnamed: 0', 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis= 1, inplace=True)


#df["HOTEL_SCORE"]= pd.qcut(df["HOTEL_COUNT"], 5, labels= [1, 2, 3, 4, 5])
#df["TOURISM_SCORE"]= pd.qcut(df["TOURISM_COUNT"], 5, labels= [1, 2, 3, 4, 5])
#df["HEALTH_SCORE"]= pd.qcut(df["HEALTH_COUNT"], 5, labels= [1, 2, 3, 4, 5])
#df["AUTOPARK_SCORE"]= pd.qcut(df["AUTOPARK_COUNT"], 5, labels= [1, 2, 3, 4, 5])
#df["GASSTATION_COUNT"]= pd.qcut(df["GASSTATION_COUNT"], 5, labels= [1, 2, 3, 4, 5])
#df["PARK_SCORE"]= pd.qcut(df["PARK_COUNT"], 5, labels= [1, 2, 3, 4, 5])

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

df[df == np.inf] = np.nan

#simpleimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#df_ng = simpleimputer.fit_transform(df)

for col in df.columns:
    if df[col].isnull().any():
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)

#df.describe().T

### CONVERTING NAN AND INF VALUES ###

#def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    #dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    #return dataframe

#ohe_cols = [col for col in df_ng.columns if 30 >= df_ng[col].nunique() > 2]

#for col in ohe_cols:
    #df_ng = one_hot_encoder(df_ng, [col])


### SCALING ###

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


### K-MEANS ###

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(scaled_data)
elbow.show()

kmeans = KMeans(n_clusters=4).fit(scaled_data)

clusters_kmeans = kmeans.labels_

data["cluster"] = clusters_kmeans

data["cluster"] = data["cluster"] + 1

data['cluster'].value_counts()

#data.groupby(['cluster']).agg(
            #AVG_NUM_VEHICLES=('AVG_NUM_VEHICLES_HOURLY', 'mean'),
            #TOTAL_NUM_VEHICLES=('TOTAL_NUM_VEHICLES_HOURLY', 'mean'),
            #AVG_TRAFFIC_DENSITY=('AVG_TRAFFIC_DENSITY_HOURLY', 'mean'),
            #HOTEL_COUNT=('HOTEL_COUNT', 'mean'),
            #TOURISM_COUNT=('TOURISM_COUNT', 'mean'),
            #HEALTH_COUNT=('HEALTH_COUNT', 'mean'),
            #AUTOPARK_COUNT=('AUTOPARK_COUNT', 'mean')).reset_index()

##############################
#### SECOND MODEL ############

df_cluster_5 = df[df['cluster'] == 4]

#def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    #dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    #return dataframe

#ohe_cols = [col for col in df_cluster_5.columns if 30 >= df_cluster_5[col].nunique() > 2]

#for col in ohe_cols:
    #df_cluster_5 = one_hot_encoder(df_cluster_5, [col])


#bool_cols = [col for col in df_cluster_5.columns if df_cluster_5[col].dtype == bool]

#for col in bool_cols:
    #df_cluster_5[col] = df_cluster_5[col].astype(int)

df_cluster_5_ng = df_cluster_5.drop(columns= ['Unnamed: 0', 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis = 1)

df_cluster_5_ng[df_cluster_5_ng == np.inf] = np.nan

#simpleimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#df_cluster_5_ng = simpleimputer.fit_transform(df_cluster_5_ng)

for col in df_cluster_5_ng.columns:
    if df_cluster_5_ng[col].isnull().any():
        mean_value = df_cluster_5_ng[col].mean()
        df_cluster_5_ng[col].fillna(mean_value, inplace=True)


### SCALING ###

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_cluster_5_ng)

kmeans = KMeans()

kmeans = KMeans(n_clusters=3).fit(scaled_data)

clusters_kmeans = kmeans.labels_

df_cluster_5["cluster"] = clusters_kmeans

df_cluster_5["cluster"] = df_cluster_5["cluster"] + 1

df_cluster_5["cluster"].value_counts()

#df_cluster_5.groupby(['cluster']).agg(
            #AVG_NUM_VEHICLES=('AVG_NUM_VEHICLES_HOURLY', 'mean'),
            #TOTAL_NUM_VEHICLES=('TOTAL_NUM_VEHICLES_HOURLY', 'mean'),
            #AVG_TRAFFIC_DENSITY=('AVG_TRAFFIC_DENSITY_HOURLY', 'mean'),
            #HOTEL_COUNT=('HOTEL_COUNT', 'mean'),
            #TOURISM_COUNT=('TOURISM_COUNT', 'mean'),
            #HEALTH_COUNT=('HEALTH_COUNT', 'mean'),
            #AUTOPARK_COUNT=('AUTOPARK_COUNT', 'mean')).reset_index()


df_cluster_5.to_csv("clustered_final.csv")

### CLUSTER VISUALIZATION ###

df = pd.read_csv('clustered_final.csv')

df = df[df['cluster'] == 3]

m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)

for index, row in df.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('cluster_unknown.html')


#############################
### PCA #####################

pca = PCA(n_components = 6)
df_PCA = pca.fit_transform(scaled_data)

print("Variance Ratio", pca.explained_variance_ratio_)
print("Sum:", sum(pca.explained_variance_ratio_))

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(whiten = True).fit(scaled_data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Clusters')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# TODO: Model to be created.

