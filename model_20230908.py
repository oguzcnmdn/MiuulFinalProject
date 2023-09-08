import joblib
import joblib
import matplotlib.pyplot as plt

plt.matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
import math
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import folium

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("Miuul_Project/dataset/updated_traffic_density_202307 (2).csv")

df = pd.read_csv("Miuul_Project/dataset/final_traffic_data_son.csv")

df1 = df.drop(columns=['GEOHASH', "Unnamed: 0"])

#################### 0 a bölme durumunda sonsuzluk sorunu

from sklearn.impute import SimpleImputer

df1.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
df_model = imputer.fit_transform(df1)

# Doldurulmuş veriyi bir DataFrame'e geri döndürelim
df_model = pd.DataFrame(df_model, columns=df1.columns)



########################   Data Preprocessing & Feature Engineering  ####################

def grab_col_names(dataframe, cat_th=13, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_model, cat_th=13, car_th=20)


######################################
# Aykırı Değer Analizi
######################################
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


for col in num_cols:
    print(col, check_outlier(df_model, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df_model, col)

# aykırı değer kontrol
for col in num_cols:
    print(col, check_outlier(df_model, col))


# df_model = df_model.drop(columns=["LATITUDE", "LONGITUDE"])


######## encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df_model.columns if 30 >= df_model[col].nunique() > 2]

for col in ohe_cols:
    df_model = one_hot_encoder(df_model, [col])

bool_cols = [col for col in df_model.columns if df_model[col].dtype == bool]

for col in bool_cols:
    df_model[col] = df_model[col].astype(int)

for col in df_model.columns:
    if df_model[col].isnull().any():
        mean_value = df_model[col].mean()
        df_model[col].fillna(mean_value, inplace=True)

df_model = df_model.drop(columns=["LATITUDE", "LONGITUDE"])

### SCALING ###

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_model)

#####K-MEANS MODELİ UYGULADIK

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

###############################
# Optimum Küme Sayısının Belirlenmesi
################################

### elbow dirsek yönt

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(scaled_data)
elbow.show()

elbow.elbow_value_  ## 5

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(scaled_data)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
scaled_data[0:5]

clusters_kmeans = kmeans.labels_

df_model["KMEANS_CLUSTER"] = clusters_kmeans

df_model["KMEANS_CLUSTER"] = df_model["KMEANS_CLUSTER"] + 1

df_model["KMEANS_CLUSTER"].value_counts()

df_model.groupby("KMEANS_CLUSTER").agg(["count", "mean", "median"])

df_model.to_csv("KMEANS_CLUSTER_1.csv")
df_model.isna().sum().sum()

df_5 = df_model[df_model["KMEANS_CLUSTER"] == 5]
df_4 = df_model[df_model["KMEANS_CLUSTER"] == 4]
df_3 = df_model[df_model["KMEANS_CLUSTER"] == 3]
df_2 = df_model[df_model["KMEANS_CLUSTER"] == 2]
df_1 = df_model[df_model["KMEANS_CLUSTER"] == 1]

df_5.describe().T

# 'cluster' sütunundaki tüm değerleri 'KMEANS_CLUSTER' ile değiştir
# df_model.rename(columns={'cluster': 'KMEANS_CLUSTER'}, inplace=True)


##########################HARİTA-1 ##########################

m = folium.Map(location=[df_4['LATITUDE'].mean(), df_4['LONGITUDE'].mean()], zoom_start=10)

for index, row in df_4.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('tahmin4.html')
