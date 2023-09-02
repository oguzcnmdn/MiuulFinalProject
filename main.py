import geohash2
import numpy as np
import pandas as pd
import requests
import folium
import geopy.distance
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import math
from geohash2 import encode
import geohash2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
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

########################################################


geolocator = Nominatim(user_agent="geo_app")

def get_location(lat, lon):
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location:
        return location.raw.get('display_name', 'Unknown')
    return 'Unknown'

df['Location'] = df.apply(lambda row: get_location(row['LATITUDE'], row['LONGITUDE']), axis=1)


########################################################
########################################################

unique_coordinates_df = df.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])

unique_latitudes = unique_coordinates_df['LATITUDE'].tolist()
unique_longitudes = unique_coordinates_df['LONGITUDE'].tolist()

len(unique_latitudes)
len(unique_longitudes)

new_df = pd.DataFrame({'LATITUDE': unique_latitudes, 'LONGITUDE': unique_longitudes})

new_df.nunique()

filtered_data = df.loc[df['GEOHASH'].str.contains('sxk3k', na=False)]

data = pd.DataFrame({
    'LATITUDE': unique_latitudes,
    'LONGITUDE': unique_longitudes
})

m = folium.Map(location=[data['LATITUDE'].mean(), data['LONGITUDE'].mean()], zoom_start=10)

for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

m.save('map.html')


#######################################
#######################################

def add_distance_fatures(df):

    df['START_LAT'] = df['LATITUDE']
    df['START_LON'] = df['LONGITUDE']
    df['END_LAT'] = df['LATITUDE'].shift(-1)
    df['END_LON'] = df['LONGITUDE'].shift(-1)

    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Yeryüzü ortalama yarıçapı (km)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    df['DISTANCE_KM'] = df.apply(
        lambda row: calculate_distance(row['START_LAT'], row['START_LON'], row['END_LAT'], row['END_LON']), axis=1)

    interval = 20

    df['DISTANCE_INTERVAL'] = (df['DISTANCE_KM'] // interval) * interval

    return df

df = add_distance_fatures(df)

df.head()

#######################################
#######################################

def add_time_features(df):

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df['HOUR_N'] = df['DATE_TIME'].dt.hour

    def date_month_year_day_names_time(date):
        year = date.year
        month_number = date.month
        day_number = date.day
        time = date.strftime('%H:%M:%S')

        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        month_name = month_names[month_number - 1]

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[date.weekday()]

        return time, day_name, day_number, month_name, year

    df['HOUR_D'], df['DAY'], df['DAY_NUMBER'], df['MONTH'], df['YEAR'] = zip(
        *df['DATE_TIME'].apply(date_month_year_day_names_time))

    df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in ['December', 'January', 'February'] else
    'Spring' if x in ['March', 'April', 'May'] else
    'Summer' if x in ['June', 'July', 'August'] else
    'Fall')

    df['DAY_INTERVAL'] = df['DAY'].apply(lambda x: 'Weekends' if x in ['Saturday', 'Sunday']
    else 'Weekdays')

    time_bins = [0, 5, 9, 12, 15, 18, 21, 24]
    time_labels = ['Late Night', 'Morning', 'Late Morning', 'Afternoon', 'Late Afternoon', 'Evening', 'Night']

    df['TIME_INTERVAL_N'] = pd.cut(df['HOUR'], bins=time_bins, labels=time_labels, include_lowest=True)

    time_bins = [0, 2, 5, 7, 10, 13, 15, 17, 20, 24]
    time_labels = ['Midnight 24:00-01:59', 'Early Morning 02:00-04:59','Morning 05:00-06:59',
                   'Morning Rush Hour 07:00-09:59','Late Morning 10:00-12:59',
                   'Afternoon 13:00-14:59', 'Late Afternoon 15:00-16:59',
                   'Evening Rush Hour 17:00-19:59', 'Night 20:00-23:59']

    df['TIME_INTERVAL_D'] = pd.cut(df['HOUR'], bins=time_bins, labels=time_labels, include_lowest=True)

    return df

df = add_time_features(df)


#######################################
#######################################

class SummaryDataFrame:
    def __init__(self, df):
        self.df = df
        self.df['DATE_TIME'] = pd.to_datetime(self.df['DATE_TIME'])
        self.df['MONTH'] = self.df['DATE_TIME'].dt.month  # Extract the month
        self.df['YEAR'] = self.df['DATE_TIME'].dt.year  # Extract the year

    def find_hourly_statistics(self):
        result_df = self.df.groupby(['GEOHASH']).agg(
            AVG_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    # AYLIK ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION
    def find_monthly_statistics(self):
        result_df = self.df.groupby(['GEOHASH', 'MONTH']).agg(
            AVG_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    # YILLIK ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION
    def find_yearly_statistics(self):
        result_df = self.df.groupby(['GEOHASH', 'YEAR']).agg(
            AVG_NUM_VEHICLES_YEARLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_YEARLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def create_summary_dataframe(self):
        hourly_df = self.find_hourly_statistics()
        monthly_df = self.find_monthly_statistics()
        yearly_df = self.find_yearly_statistics()

        merged_df = hourly_df.merge(monthly_df, on=['GEOHASH']).merge(yearly_df, on=['GEOHASH'])

        return merged_df


# Usage
summary_df = SummaryDataFrame(df)
new_df = summary_df.create_summary_dataframe()

df['TIME_INTERVAL_N'].value_counts()

# HAFTALIK ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

# MEVSIMLIK ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

# GUNUN BELLI ARALIKLARINDA (MORNING, LATE MORNING, etc.) ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

     # IS GIRISI VE IS CIKISI DA EKLENECEK.

# OZEL GUNLER (MILLI BAYRAM, DINI BAYRAM, etc.) ARALIKLARINDA ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

# HAFTASONU ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

# HAFTAICI ORTALAMA VE TOPLAM DEGERLER ICIN FEATURE PRODUCTION

#######################################
#######################################

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

#######################################
#######################################


def calculate_average_num_vehicles(df, start_hour, end_hour):

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df['HOUR'] = df['DATE_TIME'].dt.hour

    filtered_df = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]

    mean_df = filtered_df.groupby(['GEOHASH'])['NUMBER_OF_VEHICLES'].mean().reset_index()

    sum_df =  filtered_df.groupby(['GEOHASH'])['NUMBER_OF_VEHICLES'].sum().reset_index()

    merged_df = mean_df.merge(sum_df, on='GEOHASH', suffixes=(f'_MEAN_{start_hour}_{end_hour}', f'_SUM_{start_hour}_{end_hour}'))

    return merged_df

ds = calculate_average_num_vehicles(df, start_hour=0, end_hour=6)
ds = calculate_average_num_vehicles(df, start_hour=6, end_hour=12)
ds = calculate_average_num_vehicles(df, start_hour=16, end_hour=18)



def calculate_average_num_vehicles(df, start_hour, end_hour):

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['MONTH'] = df['DATE_TIME'].dt.month
    df['YEAR'] = df['DATE_TIME'].dt.year

    df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                        'Spring' if x in [3, 4, 5] else
                                        'Summer' if x in [6, 7, 8] else
                                        'Fall')

    filtered_df = df[(df['HOUR'] >= start_hour) & (df['HOUR'] <= end_hour)]

    result_df = filtered_df.groupby(['GEOHASH']).agg(
        AVG_NUM_VEHICLES_HOUR=('NUMBER_OF_VEHICLES', 'mean'),
        TOTAL_NUM_VEHICLES_HOUR=('NUMBER_OF_VEHICLES', 'sum'),
        AVG_NUM_VEHICLES_MONTH=('NUMBER_OF_VEHICLES', 'mean'),
        TOTAL_NUM_VEHICLES_MONTH=('NUMBER_OF_VEHICLES', 'sum'),
        AVG_NUM_VEHICLES_YEAR=('NUMBER_OF_VEHICLES', 'mean'),
        TOTAL_NUM_VEHICLES_YEAR=('NUMBER_OF_VEHICLES', 'sum'),
        AVG_NUM_VEHICLES_SEASON=('NUMBER_OF_VEHICLES', 'mean'),
        TOTAL_NUM_VEHICLES_SEASON=('NUMBER_OF_VEHICLES', 'sum')
    ).reset_index()

    return result_df


df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

df['HOUR'] = df['DATE_TIME'].dt.hour
df['MONTH'] = df['DATE_TIME'].dt.month
df['YEAR'] = df['DATE_TIME'].dt.year

df['SEASON'] = df['MONTH'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
'Spring' if x in [3, 4, 5] else
'Summer' if x in [6, 7, 8] else
'Fall')

x = [1,2,3,4,5,6,7,8,9,10,11,12]
for i in x:
    x_df = df[df['MONTH'] == i]
    mean_df = x_df.groupby(['GEOHASH', 'MONTH'])['NUMBER_OF_VEHICLES'].mean().reset_index()

    print(mean_df)

mean_df = df.groupby(['GEOHASH', 'MONTH'])['NUMBER_OF_VEHICLES'].mean().reset_index()

mean_df[mean_df['MONTH'] == 2]


x = df.drop_duplicates(subset='GEOHASH', keep= 'first')

y = x['GEOHASH'].drop_duplicates(keep= 'first').reset_index()

geohash_list = y['GEOHASH'].to_list()

perform_EDA(df)

#################################

df['TRAFFIC_DENSITY'] = (df["NUMBER_OF_VEHICLES"] / df["DISTANCE_KM"]) * df["AVERAGE_SPEED"]



##########################################################

df_park = pd.read_csv('sightseeing_places_ISTANBUL.csv')


latitude = df_park["LATITUDE"]
longitude = df_park["LONGITUDE"]

df_park["LATITUDE"] = df_park["LATITUDE"].astype(float)
df_park["LONGITUDE"] = df_park["LONGITUDE"].astype(float)

###  Enlem ve boylamı Geohash koduna çevir: ######
df_park["GEOHASH"] = ""
for index, row in df_park.iterrows():
    latitude = row["LATITUDE"]
    longitude = row["LONGITUDE"]
    geohash_code = geohash2.encode(latitude, longitude)
    df_park.at[index, "GEOHASH"] = geohash_code

print(df_park["GEOHASH"])


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Dünya yarıçapı (km)

    # Koordinatları float türüne dönüştürün
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Örnek trafik ve turistik geohash listeleri
turistik_geohash_list = df_park["GEOHASH"].tolist()
trafik_geohash_list = ds["GEOHASH"].tolist()

# Trafik ve turistik geohash değerlerini kullanarak turistik yer sayısını hesaplayan fonksiyon
def count_tourist_places_near_traffic(trafik_geohash_list, turistik_geohash_list):
    result = {}

    for trafik in trafik_geohash_list:
        count = 0  # Her yol için turistik yer sayısı sıfırlanır
        lat1, lon1 = geohash2.decode(trafik)  # Trafik yolunun koordinatları

        for turistik in turistik_geohash_list:
            lat2, lon2 = geohash2.decode(turistik)  # Turistik yerin koordinatları

            # İki nokta arasındaki mesafeyi hesapla ve 20 km'den küçükse say
            if haversine(lat1, lon1, lat2, lon2) <= 20:
                count += 1

        result[trafik] = count

    return result


# Fonksiyonu çağırarak sonuçları elde etme
sonuclar = count_tourist_places_near_traffic(trafik_geohash_list, turistik_geohash_list)
sonuclar_df = pd.DataFrame(list(sonuclar.items()), columns=["GEOHASH", "T_COUNT"])
trafik_df = pd.merge(ds, sonuclar_df, on="GEOHASH", how="left")

trafik_df.sort_values(by='T_COUNT', ascending=False)

trafik_df[trafik_df['T_COUNT'] >= 40].value_counts()