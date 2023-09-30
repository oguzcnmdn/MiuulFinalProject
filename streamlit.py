import streamlit as st
from streamlit_option_menu import option_menu
import requests
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


st.set_page_config(
    page_title='E-Charge Finder',
    layout="wide",
    page_icon="⚡")

page_bg_img = """
<style>
[data-testid="stSidebar"] {
background-image: url("https://unbox.ph/wp-content/uploads/2021/06/electric-car-charging-station-price-scaled-1-e1622541876593.jpg");
background-size: cover;
background-position: 40% center;
}
</style>
"""

st.markdown('<div style="text-align: center;">' + page_bg_img+ '</div>',unsafe_allow_html=True)


with st.sidebar:

    selected = option_menu("E-Charge Finder ", ["Summary", "Closest Locations", "Contacts"],
                           icons=['info-square','map', 'file'], menu_icon="play",
                           default_index=0,
                             styles={
                                "container": {"padding": "120px", "background-color": "rgba(169, 169, 169, 0.5)"},
                                "icon": {"color": "black", "font-size": "25px"},
                                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#1E90FF"},
                            }
                           )


if selected == "Summary":

    data = pd.read_csv('datasets/final_traffic_data_C.csv')

    df = data.copy()

    df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

    st.title("Dataset Overlook")
    st.write(df.head())

    def create_unique_coordinates_dataframe(dataframe):
        unique_coordinates_dataframe = dataframe.drop_duplicates(subset='GEOHASH')

        unique_geohash = unique_coordinates_dataframe['GEOHASH'].tolist()
        unique_latitudes = unique_coordinates_dataframe['LATITUDE'].tolist()
        unique_longitudes = unique_coordinates_dataframe['LONGITUDE'].tolist()

        dataframe = pd.DataFrame({'GEOHASH': unique_geohash, 'LATITUDE': unique_latitudes, 'LONGITUDE': unique_longitudes})

        return dataframe

    df_unique_geohash_locations = create_unique_coordinates_dataframe(df)

    #st.map(df_unique_geohash_locations, latitude='LATITUDE', longitude='LONGITUDE', color = '#0044ff')

#######################################################################################################################
######################################## IMPORTANT LOCATIONS VISUALIZATION ############################################
#######################################################################################################################


    df_touristic = pd.read_csv('datasets/location_datasets/touristics.csv')
    df_parks = pd.read_csv('datasets/location_datasets/parks.csv')
    df_health = pd.read_csv('datasets/location_datasets/health.csv')
    df_hotels = pd.read_csv('datasets/location_datasets/hotels.csv')
    df_gasstations = pd.read_csv('datasets/location_datasets/gas_stations.csv')
    df_autoparks = pd.read_csv('datasets/location_datasets/autoparks.csv')

    df_touristic['Type'] = "Touristic"
    df_parks['Type'] = 'Park'
    df_health['Type'] = 'Health'
    df_hotels['Type'] = 'Hotel'
    df_gasstations['Type'] = 'Gas Station'
    df_autoparks['Type'] = 'Autopark'

    combined_df = pd.concat([df_touristic, df_parks, df_health,
                             df_hotels, df_gasstations, df_autoparks], ignore_index=True)

    combined_df['LONGITUDE'] = combined_df['LONGITUDE'].astype(float)
    combined_df['LATITUDE'] = combined_df['LATITUDE'].astype(float)

    combined_df = combined_df.dropna(subset=['LATITUDE', 'LONGITUDE'])

    type_colors = {'Touristic': '#FF0000', 'Park': '#00FF00', 'Health': '#0000FF',
                   'Hotel': '#FFFF00', 'Gas Station': '#8B00FF', 'Autopark': '#FF7F00'}

    combined_df['Color'] = combined_df['Type'].map(type_colors)

    #st.map(combined_df, latitude='LATITUDE', longitude='LONGITUDE', color = 'Color')


#######################################################################################################################
################################################# PREPARATION #########################################################
#######################################################################################################################

    df.drop(columns=[ 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis=1, inplace=True)


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


    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=31, car_th=20)


    df[df == np.inf] = np.nan

    for col in df.columns:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)

#######################################################################################################################
################################################# 1ST K-MEANS #########################################################
#######################################################################################################################

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(scaled_data)
    elbow.show()

    kmeans = KMeans(n_clusters=4).fit(scaled_data)

    clusters_kmeans = kmeans.labels_

    data["cluster"] = clusters_kmeans

    data["cluster"] = data["cluster"] + 1

    data["cluster"].value_counts()

    st.title(f'First Model (K = {4})')
    st.line_chart(elbow.k_scores_, width=500, use_container_width=True)

    cluster_lengths = data.groupby('cluster').size().reset_index(name='cluster_length')

    selected_clusters = cluster_lengths[
        (cluster_lengths['cluster_length'] >= 400) & (cluster_lengths['cluster_length'] <= 450)]

    selected_cluster_numbers = selected_clusters['cluster'].tolist()

#######################################################################################################################
################################################# 2ND K-MEANS #########################################################
#######################################################################################################################

    df_cluster_best_FM = data[data['cluster'] == selected_cluster_numbers[0]]

    #st.map(df_cluster_best_FM, latitude='LATITUDE', longitude='LONGITUDE', color = '#0044ff')


    df_cluster_best_FM_d = df_cluster_best_FM.drop(columns=['Unnamed: 0', 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis=1)

    df_cluster_best_FM_d[df_cluster_best_FM_d == np.inf] = np.nan

    for col in df_cluster_best_FM_d.columns:
        if df_cluster_best_FM_d[col].isnull().any():
            mean_value = df_cluster_best_FM_d[col].mean()
            df_cluster_best_FM_d[col].fillna(mean_value, inplace=True)

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_cluster_best_FM_d)
    kmeans = KMeans()
    elbow.fit(scaled_data)
    elbow.show()

    kmeans = KMeans(n_clusters=3).fit(scaled_data)

    clusters_kmeans = kmeans.labels_

    df_cluster_best_FM["cluster"] = clusters_kmeans

    df_cluster_best_FM["cluster"] = df_cluster_best_FM["cluster"] + 1

    df_cluster_best_FM["cluster"].value_counts()

    st.title(f'Second Model (K = {3})')
    st.line_chart(elbow.k_scores_, width=500, use_container_width=True)

    df_cluster_best_FM.to_csv("datasets/clustered_final.csv")

    df = pd.read_csv('datasets/clustered_final.csv')

    cluster_lengths = df.groupby('cluster').size().reset_index(name='cluster_length')

    selected_clusters = cluster_lengths[
        (cluster_lengths['cluster_length'] >= 190) & (cluster_lengths['cluster_length'] <= 200)]

    selected_cluster_numbers = selected_clusters['cluster'].tolist()

    df = df[df['cluster'] == selected_cluster_numbers[0]]

    df.to_csv("final_best_cluster.csv")

    df_cluster_ = df.loc[:, ["cluster", "LATITUDE", "LONGITUDE"]]

    #st.title("Demonstration on Map")

    st.map(df, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')



elif selected == 'Closest Locations':

    df_touristic = pd.read_csv('location_datasets/touristics.csv')

    df_touristic['LONGITUDE'] = df_touristic['LONGITUDE'].astype(float)
    df_touristic['LATITUDE'] = df_touristic['LATITUDE'].astype(float)

    selected_location = st.selectbox('Select a Location:', df_touristic['NAME'])


    if st.button('Find'):
        def user_input_features():
            selected_row = df_touristic[df_touristic['NAME'] == selected_location]
            latitude = selected_row['LATITUDE'].values[0]
            longitude = selected_row['LONGITUDE'].values[0]

            data_1 = {'LATITUDE': latitude,
                      'LONGITUDE': longitude}

            features = pd.DataFrame(data_1, index=[0])
            return features

        input_df = user_input_features()


        st.title("Your Location")
        st.map(input_df, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')
        df = pd.read_csv("final_best_cluster.csv")

        st.title("Closest Locations")

        df_location = df.loc[:, ["GEOHASH", "LATITUDE", "LONGITUDE", "cluster"]]
        df_location["LATITUDE_input"] = input_df["LATITUDE"]
        df_location = df_location.fillna(input_df["LATITUDE"].values[0])
        df_location["LONGITUDE_input"] = input_df["LONGITUDE"]
        df_location = df_location.fillna(input_df["LONGITUDE"].values[0])


        def haversine(lat1, lon1, lat2, lon2):

            R = 6371

            lat1 = math.radians(lat1)
            lon1 = math.radians(lon1)
            lat2 = math.radians(lat2)
            lon2 = math.radians(lon2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            return distance


        df_location['Distance'] = df_location.apply(
            lambda row: haversine(row['LATITUDE_input'], row['LONGITUDE_input'], row['LATITUDE'], row['LONGITUDE']),
            axis=1)
        df_short = df_location.sort_values(by='Distance', ascending=True).head()
        st.map(df_short, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')



else:

    col1, col2, = st.columns(2)

    with col1:
        st.markdown(
            """
            <a href="https://www.miuul.com/" target="_blank">
                <img src="https://miuul.com/image/theme/logo-dark.png" alt="Miuul" width="360" height="240">
            </a>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <a href="https://bootcamp.veribilimiokulu.com/bootcamp-programlari/veri-bilimci-yetistirme-programi/" target="_blank">
                <img src="https://www.veribilimiokulu.com/wp-content/uploads/2020/12/veribilimiokulu_logo-crop.png" alt="Veri Bilimi Okulu" width="360" height="240">
            </a>
            """,
            unsafe_allow_html=True
        )

    col1, = st.columns(1)

    with col1:
        video_url = "https://www.youtube.com/watch?v=Ww9M0WJfGN8"
        st.video(video_url)

    st.title("Contributors")


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/oguzcnmdn/" target="_blank">
                <img src="https://media.licdn.com/dms/image/D4D03AQH-wM9z39yE2w/profile-displayphoto-shrink_400_400/0/1681664633623?e=1700092800&v=beta&t=qyA9DKzKU6DsffTX45UEnJd1cPeui7k9nZSCiIP9hCA" alt="Oğuzcan Maden" width="160" height="160">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style='font-family: "Pacifico"; font-size: 20px; font-weight: bold;'>
                Oğuzcan Maden
            </p>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/yaseminergun" target="_blank">
                <img src="https://media.licdn.com/dms/image/C5603AQEAUpI-UcNZ-w/profile-displayphoto-shrink_400_400/0/1663076428087?e=1700092800&v=beta&t=c0pAQt6hyBjywDp031L6FUsl4XNICFBZIfRwveUxQ-U" alt="Yasemin Ergün" width="160" height="160">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style='font-family: "Pacifico"; font-size: 20px; font-weight: bold;'>
                Yasemin Ergün
            </p>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <a href="http://www.linkedin.com/in/fatmayagmurlu" target="_blank">
                <img src="https://media.licdn.com/dms/image/D4D03AQF0Jn2bBldCSg/profile-displayphoto-shrink_400_400/0/1689946874271?e=1700092800&v=beta&t=ifddJffMqPur33WcdDaDfsKB0jzNH-ZrNCAtb1I16uo" alt="Fatma Yağmurlu" width="160" height="160">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style='font-family: "Pacifico"; font-size: 20px; font-weight: bold;'>
                Fatma Yağmurlu
            </p>
            """,
            unsafe_allow_html=True
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col2:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/ozan-bahar/" target="_blank">
                <img src="https://media.licdn.com/dms/image/C4D03AQEDhD4TUv1qtA/profile-displayphoto-shrink_400_400/0/1651916873528?e=1700092800&v=beta&t=oSGrRMFJokST7l3xgfoiLuDZn3J-2DUGqDGnuMEGhTk" alt="Ozan Bahar" width="160" height="160">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style='font-family: "Pacifico"; font-size: 20px; font-weight: bold;'>
                Ozan Bahar
            </p>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/cenk-bayender-98a285130" target="_blank">
                <img src="https://media.licdn.com/dms/image/C4D03AQHsrawgOl69JA/profile-displayphoto-shrink_400_400/0/1605110493801?e=1700092800&v=beta&t=-39i3D_qFPdu9gsZrx8MwaPM9OB28eXKnWJbWqwIzYA" alt="Cenk Bayender" width="160" height="160">
            </a>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <p style='font-family: "Pacifico"; font-size: 20px; font-weight: bold;'>
                Cenk Bayender
            </p>
            """,
            unsafe_allow_html=True
        )


