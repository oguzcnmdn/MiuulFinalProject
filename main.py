import math
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import folium
import geopy.distance
from geopy.geocoders import Nominatim
import geopandas as gpd
import geohash2
from geohash2 import encode
from SIHelper import Jerzy, Abraham


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"Failed to download {filename}.")

file_info = [
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/287e7fc9-6d92-4019-ac58-ff6bca6e6151/download/traffic_density_202207.csv", "traffic_density_202207.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/acd85951-6d23-4b50-bac6-d941f92af1ad/download/traffic_density_202208.csv", "traffic_density_202208.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/a5da03fe-4a89-493b-ae60-aeb132511be9/download/traffic_density_202209.csv", "traffic_density_202209.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/72183a60-d47f-4dc9-b1dc-fced0649dcf5/download/traffic_density_202210.csv", "traffic_density_202210.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/7f463362-a580-41d9-a86a-a542818e7542/download/traffic_density_202211.csv", "traffic_density_202211.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/dc788908-2b75-434f-9f3f-ef82ff33a158/download/traffic_density_202212.csv", "traffic_density_202212.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/42fa7a5f-29f1-4b38-9dfa-ac7c8fe3c77d/download/traffic_density_202301.csv", "traffic_density_202301.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/366befd8-defd-4f79-a3d2-0e7948c649ff/download/traffic_density_202302.csv", "traffic_density_202302.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/6a60b03a-bf25-4575-9dce-e21fe0e04e77/download/traffic_density_202303.csv", "traffic_density_202303.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/ce65562e-0d17-4d7e-8090-9484990a8f2b/download/traffic_density_202304.csv", "traffic_density_202304.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/d0a71c11-47d2-4f98-8745-c9446b10bf18/download/traffic_density_202305.csv", "traffic_density_202305.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/a99913df-dccc-4b7d-b6e3-963ccb5d27b1/download/traffic_density_202306.csv", "traffic_density_202306.csv"),
    ("https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/3de18c1e-57c0-4493-9b75-5a896edae0ff/download/traffic_density_202307.csv", "traffic_density_202307.csv")
]

for url, filename in file_info:
    download_file(url, filename)


def concatenate_csv_files(filenames, output_filename):
    dfs = []

    for filename in filenames:
        df = pd.read_csv(filename)

        if "2022" in filename:
            df["LATITUDE"], df["LONGITUDE"] = df["LONGITUDE"], df["LATITUDE"]

        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    concatenated_df.to_csv(output_filename, index=False)
    print(f"Concatenated data saved to {output_filename}.")


input_filenames = [filename for url, filename in file_info]

output_filename = "concatenated_traffic_data.csv"

concatenate_csv_files(input_filenames, output_filename)

data = pd.read_csv(output_filename)

df = data.copy()


###################################################################################################
############################    FEATURE ENGINEERING   #############################################
###################################################################################################

abraham = Abraham()
jerzy = Jerzy(df)

df = abraham.add_distance_features(df)
df = abraham.add_time_features(df)
df_base = abraham.create_unique_coordinates_dataframe(df)

df_hour = jerzy.find_based_statistics()

def find_and_merge_statistics(jerzy, group_column):
    df = jerzy.find_custom_statistics_C(group_column=group_column)
    return df

def merge_all_statistics(jerzy, group_columns):
    merged_df = None
    for column in group_columns:
        df = find_and_merge_statistics(jerzy, group_column=column)
        if merged_df is None:
            merged_df = jerzy.find_based_statistics()
        else:
            merged_df = merged_df.merge(df, on='GEOHASH')
    return merged_df

group_columns = df.columns[11:]

final_df = merge_all_statistics(jerzy, group_columns)


###################################################################################################
###################################################################################################
###################################################################################################

df_hotel = pd.read_excel('location_datasets/hotels.xlsx')
df_tourism = pd.read_csv("location_datasets/touristics.csv")
df_health = pd.read_excel("location_datasets/health.xlsx")
df_autopark = pd.read_excel("location_datasets/autoparks.xlsx")
df_gasstation = pd.read_csv("location_datasets/gas_stations.csv")
df_park = pd.read_csv("location_datasets/parks.csv")


def find_lon_lan_dataframe(df):

    latitude = df["LATITUDE"]
    longitude = df["LONGITUDE"]

    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)

    df["GEOHASH"] = ""
    for index, row in df.iterrows():
        latitude = row["LATITUDE"]
        longitude = row["LONGITUDE"]
        geohash_code = geohash2.encode(latitude, longitude)
        df.at[index, "GEOHASH"] = geohash_code

    return df

df_hotel = find_lon_lan_dataframe(df_hotel)
df_tourism = find_lon_lan_dataframe(df_tourism)
df_health = find_lon_lan_dataframe(df_health)
df_autopark = find_lon_lan_dataframe(df_autopark)

df_gasstation.columns = ['NAME', 'LATITUDE', 'LONGITUDE', 'NEIGHBORHOOD_NAME']

df_gasstation = find_lon_lan_dataframe(df_gasstation)
df_park = find_lon_lan_dataframe(df_park)


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2

    R = 6371

    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def geohash_to_list(df1, final_df):

    df1_geohash_list = df1["GEOHASH"].tolist()
    final_geohash_list = final_df["GEOHASH"].tolist()

    return df1_geohash_list, final_geohash_list

hotel_g_list, final_g_list = geohash_to_list(df_hotel, final_df)
tourism_g_list, final_g_list = geohash_to_list(df_tourism, final_df)
health_g_list, final_g_list = geohash_to_list(df_health, final_df)
autopark_g_list, final_g_list = geohash_to_list(df_autopark, final_df)
gasstation_g_list, final_g_list = geohash_to_list(df_gasstation, final_df)
park_g_list, final_g_list = geohash_to_list(df_park, final_df)



unique_list = []

def unique(list1):
    # initialize a null list

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

unique(final_g_list)

def count_places_near_traffic(final_g_list, place_g_list):
    result = {}

    for i in final_g_list:
        count = 0
        lat1, lon1 = geohash2.decode(i)

        for place in place_g_list:
            lat2, lon2 = geohash2.decode(place)

            if haversine(lat1, lon1, lat2, lon2) <= 20:
                count += 1

        result[i] = count

    return result


hotel_result = count_places_near_traffic(unique_list, hotel_g_list)
tourism_result = count_places_near_traffic(unique_list, tourism_g_list)
health_result = count_places_near_traffic(unique_list, health_g_list)
autopark_result = count_places_near_traffic(unique_list, autopark_g_list)
gasstation_result = count_places_near_traffic(unique_list, gasstation_g_list)
park_result = count_places_near_traffic(unique_list, park_g_list)

hotel_result_df = pd.DataFrame(list(hotel_result.items()), columns=["GEOHASH", "HOTEL_COUNT"])
tourism_result_df = pd.DataFrame(list(tourism_result.items()), columns=["GEOHASH", "TOURISM_COUNT"])
health_result_df = pd.DataFrame(list(health_result.items()), columns=["GEOHASH", "HEALTH_COUNT"])
autopark_result_df = pd.DataFrame(list(autopark_result.items()), columns=["GEOHASH", "AUTOPARK_COUNT"])
gasstation_result_df = pd.DataFrame(list(gasstation_result.items()), columns=["GEOHASH", "GASSTATION_COUNT"])
park_result_df = pd.DataFrame(list(park_result.items()), columns=["GEOHASH", "PARK_COUNT"])


final_df = pd.merge(final_df, hotel_result_df, on="GEOHASH", how="left")
final_df = pd.merge(final_df, tourism_result_df, on="GEOHASH", how="left")
final_df = pd.merge(final_df, health_result_df, on="GEOHASH", how="left")
final_df = pd.merge(final_df, autopark_result_df, on="GEOHASH", how="left")
final_df = pd.merge(final_df, gasstation_result_df, on="GEOHASH", how="left")
final_df = pd.merge(final_df, park_result_df, on="GEOHASH", how="left")

final_df.to_csv('final_traffic_data_C.csv')

###################################################################################################
###################################################################################################
###################################################################################################

def find_missing_data_days(start_date, end_date, date_column):
    empty_year = pd.date_range(start=start_date, end=end_date, periods=(end_date - start_date).days + 1, normalize=True)
    all_days = set(empty_year)
    observation_days = set([i.normalize() for i in date_column])
    missing_days = list(all_days.difference(observation_days))
    missing_days.sort()
    missing_days_count = len(missing_days)
    missing_days_percent = (1 - (len(observation_days) / len(all_days))) * 100

    print(f"Days with Missing Data: {missing_days_count} (%{missing_days_percent:.1f})")
    for day in missing_days:
        print(day.date())


###################################################################################################
############################   MODEL CREATION   ###################################################
###################################################################################################


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

df[df["cluster"]==5]

df.to_csv("clusters.csv")


df_ = pd.read_csv('updated_concatenated_traffic_data.csv')


unique_coordinates_df_ = df_.drop_duplicates(subset="GEOHASH")

unique_geohash = unique_coordinates_df_['GEOHASH'].tolist()
unique_latitudes = unique_coordinates_df_['LATITUDE'].tolist()
unique_longitudes = unique_coordinates_df_['LONGITUDE'].tolist()

df_with_lat_lon = pd.DataFrame({'GEOHASH': unique_geohash, 'LATITUDE': unique_latitudes, 'LONGITUDE': unique_longitudes})



df = df.merge(df_with_lat_lon,on="GEOHASH")

df_cluster_5 = df[df['cluster'] == 2]


m = folium.Map(location=[df_cluster_5['LATITUDE'].mean(), df_cluster_5['LONGITUDE'].mean()], zoom_start=10)

for index, row in df_cluster_5.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('tahmin2.html')


### PCA ###

pca = PCA(n_components = 8)
pca.fit(df)

x_pca = pca.transform(df)

print("Variance Ratio", pca.explained_variance_ratio_)
print("Sum:", sum(pca.explained_variance_ratio_))

# TODO: Model kurulacak.


###################################################################################################
###################################################################################################
###################################################################################################


geolocator = Nominatim(user_agent="geo_app")

def get_location(lat, lon):
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location:
        return location.raw.get('display_name', 'Unknown')
    return 'Unknown'

df['Location'] = df.apply(lambda row: get_location(row['LATITUDE'], row['LONGITUDE']), axis=1)