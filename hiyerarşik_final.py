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

dff = pd.read_csv("Miuul_Project/dataset/final_traffic_data_son.csv")

dff1 = dff.drop(columns=['GEOHASH', "Unnamed: 0"])

#################### 0 a bölme durumunda sonsuzluk sorunu

from sklearn.impute import SimpleImputer

dff1.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
df_model1 = imputer.fit_transform(dff1)

# Doldurulmuş veriyi bir DataFrame'e geri döndürelim
df_model1 = pd.DataFrame(df_model1, columns=dff1.columns)


########################   Data Preprocessing & Feature Engineering ####################

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


cat_cols, num_cols, cat_but_car = grab_col_names(df_model1, cat_th=13, car_th=20)


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
    print(col, check_outlier(df_model1, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df_model1, col)

# aykırı değer kontrol
for col in num_cols:
    print(col, check_outlier(df_model1, col))


# df_model = df_model.drop(columns=["LATITUDE", "LONGITUDE"])


######## encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df_model1.columns if 30 >= df_model1[col].nunique() > 2]

for col in ohe_cols:
    df_model1 = one_hot_encoder(df_model1, [col])

bool_cols = [col for col in df_model1.columns if df_model1[col].dtype == bool]

for col in bool_cols:
    df_model1[col] = df_model1[col].astype(int)

for col in df_model1.columns:
    if df_model1[col].isnull().any():
        mean_value = df_model1[col].mean()
        df_model1[col].fillna(mean_value, inplace=True)

df_model1 = df_model1.drop(columns=["LATITUDE", "LONGITUDE"])

### SCALING ###

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_model1)

################################
# Hierarchical Clustering
################################

df_model1  # aldım

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_model1)

hc_average = linkage(scaled_data, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10,
           above_threshold_color='r'
           )
plt.axhline(y=0.08, color='b', linestyle='--')
plt.show()

###############################
# Optimum Küme Sayısının Belirlenmesi
################################

################################
# Kume Sayısını Belirlemek 4 olarak al  veya 5
################################

plt.figure(figsize=(25, 20))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10,
           truncate_mode="lastp",
           p=30,
           show_contracted=True)
plt.axhline(y=20, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df_model1)
model_df1 = pd.DataFrame(scaled_data, columns=df_model1.columns)
model_df1.head()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, linkage="average")
segments = hc.fit_predict(model_df1)

df2 = pd.read_csv("Miuul_Project/dataset/final_traffic_data_son.csv")

df_result = df2.drop(columns=['GEOHASH', "Unnamed: 0"])

df_result["HI_CLUSTER_NO"] = segments
df_result["HI_CLUSTER_NO"] = df_result["HI_CLUSTER_NO"] + 1
df_result["HI_CLUSTER_NO"].value_counts()

df_result.groupby("HI_CLUSTER_NO").agg({"TRAFFIC_DENSITY": ["mean", "min", "max"],
                                        "NUMBER_OF_VEHICLES": ["mean", "min", "max"],
                                        "TOURISM_COUNT": ["mean", "min", "max"],
                                        "HEALTH_COUNT": ["mean", "min", "max"],
                                        "AUTOPARK_COUNT": ["mean", "min", "max", "count"],
                                        "GASSTATION_COUNT": ["mean", "min", "max"],
                                        "PARK_COUNT": ["mean", "min", "max"],
                                        "HOTEL_COUNT": ["mean", "min", "max", "count"]})

hi_cluster_4 = df_result[df_result["HI_CLUSTER_NO"] == 4]["KMEANS_CLUSTER_NO"]

df_result["KMEANS_CLUSTER_NO"].value_counts()
df_result["HI_CLUSTER_NO"].value_counts()

########################## HARİTA-Hiyeraarşik ##########################
hi_cluster_1 = df_result[df_result["HI_CLUSTER_NO"] == 1]
hi_cluster_2 = df_result[df_result["HI_CLUSTER_NO"] == 2]
hi_cluster_3 = df_result[df_result["HI_CLUSTER_NO"] == 3]
hi_cluster_4 = df_result[df_result["HI_CLUSTER_NO"] == 4]
hi_cluster_5 = df_result[df_result["HI_CLUSTER_NO"] == 5]

############## HARİTA 4  CLUSTER İYİ GİBİ

m = folium.Map(location=[hi_cluster_4['LATITUDE'].mean(), hi_cluster_4['LONGITUDE'].mean()], zoom_start=10)

for index, row in hi_cluster_4.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('tahminhH4444.html')
