"""
This script downloads traffic density data from multiple CSV files, processes it, and performs various operations on it.

It also integrates data from different location datasets and generates additional features related to points of interest.

The script consists of the following major sections:

1. Importing necessary libraries and modules.
2. Setting display options for pandas.
3. Defining utility functions for file downloading and concatenation.
4. Downloading traffic density files from URLs.
5. Concatenating downloaded CSV files into a single DataFrame.
6. Loading and preprocessing the traffic data.
7. Initializing instances of Abraham and Jerzy classes.
8. Adding distance and time features using Abraham's methods.
9. Creating a base DataFrame with unique coordinates.
10. Finding and merging custom statistics using Jerzy's method.
11. Merging all statistics based on different group columns using Jerzy's methods.
12. Reading and preprocessing location datasets using Marco's methods.
13. Counting places near traffic points using Marco's method.
14. Merging the counts with the final traffic DataFrame.
15. Finding and printing days with missing data using a utility function.
16. Geocoding latitude and longitude coordinates to get location names.

Note: This script assumes that the necessary files and datasets are available in the specified paths.

"""

#######################################################################################################################
## 1. Importing necessary libraries and modules. ######################################################################
#######################################################################################################################

import pandas as pd
import requests
from geopy.geocoders import Nominatim
from SIHelper import Jerzy, Abraham, Marco
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################################
## 2. Setting display options for pandas. #############################################################################
#######################################################################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#######################################################################################################################
## 3. Defining utility functions for file downloading and concatenation. ##############################################
#######################################################################################################################

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

#######################################################################################################################
## 4. Downloading traffic density files from URLs. ####################################################################
#######################################################################################################################

for url, filename in file_info:
    download_file(url, filename)

#######################################################################################################################
## 5. Concatenating downloaded CSV files into a single DataFrame. #####################################################
#######################################################################################################################

def concatenate_csv_files(filenames, output_filename):
    dfs = []

    for filename in filenames:
        df = pd.read_csv(filename)

        if "2022" in filename:
            df["LATITUDE"], df["LONGITUDE"] = df["LONGITUDE"], df["LATITUDE"]

        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)

    concatenated_df.to_csv(f'dataset\{output_filename}', index=False)
    print(f"Concatenated data saved to {output_filename}.")


input_filenames = [filename for url, filename in file_info]

output_filename = "concatenated_traffic_data.csv"

concatenate_csv_files(input_filenames, output_filename)

#######################################################################################################################
## 6. Loading and preprocessing the traffic data. #####################################################################
#######################################################################################################################

data = pd.read_csv(output_filename)

df = data.copy()

#######################################################################################################################
## 7. Initializing instances of Abraham and Jerzy classes. ############################################################
#######################################################################################################################

abraham = Abraham()
jerzy = Jerzy(df)

#######################################################################################################################
## 8. Adding distance and time features using Abraham's methods. ######################################################
#######################################################################################################################

df = abraham.add_distance_features(df)
df = abraham.add_time_features(df)

#######################################################################################################################
## 9. Creating a base DataFrame with unique coordinates. ##############################################################
#######################################################################################################################

df_base = abraham.create_unique_coordinates_dataframe(df)

#######################################################################################################################
## 10. Finding and merging custom statistics using Jerzy's method. ####################################################
#######################################################################################################################

def find_and_merge_statistics(jerzy, group_column):
    dataframe = jerzy.find_custom_statistics_C(group_column=group_column)
    return dataframe

#######################################################################################################################
## 11. Merging all statistics based on different group columns using Jerzy's methods. #################################
#######################################################################################################################

def merge_all_statistics(jerzy, base_dataframe, group_columns):
    merged_dataframe = None
    for column in group_columns:
        dataframe = find_and_merge_statistics(jerzy, group_column=column)
        if merged_dataframe is None:
            merged_dataframe = base_dataframe
            dataframe_hourly = jerzy.find_based_statistics()
            merged_dataframe = merged_dataframe.merge(dataframe_hourly, on='GEOHASH')
        else:
            merged_dataframe = merged_dataframe.merge(dataframe, on='GEOHASH')
    return merged_dataframe

group_columns = df.columns[18:]

final_df = merge_all_statistics(jerzy, df_base, group_columns)

#######################################################################################################################
## 12. Reading and preprocessing location datasets using Marco's methods. #############################################
#######################################################################################################################

df_hotel = pd.read_excel('location_datasets/hotels.xlsx')
df_tourism = pd.read_csv("location_datasets/touristics.csv")
df_health = pd.read_excel("location_datasets/health.xlsx")
df_autopark = pd.read_excel("location_datasets/autoparks.xlsx")
df_gasstation = pd.read_csv("location_datasets/gas_stations.csv")
df_park = pd.read_csv("location_datasets/parks.csv")

marco = Marco()

df_hotel = marco.find_lon_lan_dataframe(df_hotel)
df_tourism = marco.find_lon_lan_dataframe(df_tourism)
df_health = marco.find_lon_lan_dataframe(df_health)
df_autopark = marco.find_lon_lan_dataframe(df_autopark)
df_gasstation = marco.find_lon_lan_dataframe(df_gasstation)
df_park = marco.find_lon_lan_dataframe(df_park)

#######################################################################################################################
## 13. Counting places near traffic points using Marco's method. ######################################################
#######################################################################################################################

hotel_result = marco.count_places_near_traffic(df_hotel,final_df)
tourism_result = marco.count_places_near_traffic(df_tourism,final_df)
health_result = marco.count_places_near_traffic(df_health,final_df)
autopark_result = marco.count_places_near_traffic(df_autopark,final_df)
gasstation_result = marco.count_places_near_traffic(df_gasstation,final_df)
park_result = marco.count_places_near_traffic(df_park,final_df)

#######################################################################################################################
## 14. Merging the counts with the final traffic DataFrame. ###########################################################
#######################################################################################################################

hotel_result_df = pd.DataFrame(list(hotel_result.items()), columns=["GEOHASH", "HOTEL_COUNT"])
tourism_result_df = pd.DataFrame(list(tourism_result.items()), columns=["GEOHASH", "TOURISM_COUNT"])
health_result_df = pd.DataFrame(list(health_result.items()), columns=["GEOHASH", "HEALTH_COUNT"])
autopark_result_df = pd.DataFrame(list(autopark_result.items()), columns=["GEOHASH", "AUTOPARK_COUNT"])
gasstation_result_df = pd.DataFrame(list(gasstation_result.items()), columns=["GEOHASH", "GASSTATION_COUNT"])
park_result_df = pd.DataFrame(list(park_result.items()), columns=["GEOHASH", "PARK_COUNT"])

final_df = final_df.merge(hotel_result_df, on='GEOHASH').\
                    merge(tourism_result_df, on='GEOHASH').\
                    merge(health_result_df, on='GEOHASH').\
                    merge(autopark_result_df, on='GEOHASH').\
                    merge(gasstation_result_df, on='GEOHASH').\
                    merge(park_result_df, on='GEOHASH')

final_df.to_csv("datasets\\final_traffic_data_C.csv", index=False)

#######################################################################################################################
## 15. Finding and printing days with missing data using a utility function. ##########################################
#######################################################################################################################

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

#######################################################################################################################
## 16. Geocoding latitude and longitude coordinates to get location names. ############################################
#######################################################################################################################

geolocator = Nominatim(user_agent="geo_app")

def get_location(lat, lon):
    location = geolocator.reverse((lat, lon), exactly_one=True)
    if location:
        return location.raw.get('display_name', 'Unknown')
    return 'Unknown'

final_df['Location'] = final_df.apply(lambda row: get_location(row['LATITUDE'], row['LONGITUDE']), axis=1)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################