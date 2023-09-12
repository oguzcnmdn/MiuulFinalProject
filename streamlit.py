import streamlit as st

import math
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


st.set_page_config(page_title="Electrical Vehicle", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Summary", "Finding a shortest route", "Contact"])


#image_url = "https://github.com/CenkBayender/MiuulFinalProject/blob/main/Logo1.jpeg"

# Görüntüyü ekleme
#image = st.image(image_url, use_column_width=True)

#st.markdown("<div style='text-align: center;'><img src='https://github.com/CenkBayender/MiuulFinalProject/blob/main/Logo1.jpeg'></div>", unsafe_allow_html=True)


#st.image("your_image.jpg", caption="Resim Açıklaması", use_column_width=True)


if page == "Summary":


    st.markdown("""
        <style>
            /* Başlık için font büyüklüğünü ve rengini ayarlayın */
            .title-text {
                font-size: 48px;
                color: #2596be; /* Başlık rengini buradan ayarlayabilirsiniz */
                text-align: center; /* Başlığı ortala */
                font-weight: bold; /* Kalın (bold) yazı tipi */
            }
        </style>
    """, unsafe_allow_html=True)

    # Başlığı belirli bir stil sınıfıyla oluşturun
    st.markdown('<p class="title-text">E-CHARGEFINDER</p>', unsafe_allow_html=True)

    #url = "https://raw.githubusercontent.com/oguzcnmdn/MiuulFinalProject/main/datasets/final_traffic_data_C.csv"
    #df = pd.read_csv(url)
    # st.write("Kulalnıcı bilgilerini ve veri setini birleştirelim.")

    #df_ = pd.read_csv("traffic_density_202207.csv")

    data = pd.read_csv('final_traffic_data_C.csv')

    df = data.copy()
    st.title("Traffic Data")
    st.write(df.head())

    df.drop(columns=['Unnamed: 0', 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis=1, inplace=True)


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

    ### SCALING ###

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)


    #def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
     #   dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
      #  return dataframe


    #ohe_cols = [col for col in df_X.columns if 30 >= df_X[col].nunique() > 2]

   # for col in ohe_cols:
   #     df_X = one_hot_encoder(df_X, [col])

   # bool_cols = [col for col in df_X.columns if df_X[col].dtype == bool]

   # for col in bool_cols:
   #     df_X[col] = df_X[col].astype(int)

    #for col in df_X.columns:
     #   if df_X[col].isnull().any():
      #      mean_value = df_X[col].mean()
      #      df_X[col].fillna(mean_value, inplace=True)

    ### SCALING ###

    #from sklearn.preprocessing import RobustScaler

   # scaler = RobustScaler()
   # scaled_data = scaler.fit_transform(df_X)

   # from sklearn.cluster import KMeans
   # from yellowbrick.cluster import KElbowVisualizer

    ### K-MEANS ###

    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(scaled_data)
    elbow.show()

    kmeans = KMeans(n_clusters=4).fit(scaled_data)

    clusters_kmeans = kmeans.labels_

    data["cluster"] = clusters_kmeans

    data["cluster"] = data["cluster"] + 1

    data["cluster"].value_counts()

    st.title(f'First Model Elbow Method K ={4}')
    st.line_chart(elbow.k_scores_, width=500, use_container_width=True) #elbow kscores çalışıyor mu

    # cluster index sabitleme

    cluster_lengths = data.groupby('cluster').size().reset_index(name='cluster_length')

    selected_clusters = cluster_lengths[
        (cluster_lengths['cluster_length'] >= 400) & (cluster_lengths['cluster_length'] <= 450)]

    selected_cluster_numbers = selected_clusters['cluster'].tolist()

    ##############################
    #### SECOND MODEL ############

    df_cluster_5 = data[data['cluster'] == selected_cluster_numbers[0]]
    df_cluster_5_ng = df_cluster_5.drop(columns=['Unnamed: 0', 'GEOHASH', 'LATITUDE', 'LONGITUDE'], axis=1)

    df_cluster_5_ng[df_cluster_5_ng == np.inf] = np.nan

    for col in df_cluster_5_ng.columns:
        if df_cluster_5_ng[col].isnull().any():
            mean_value = df_cluster_5_ng[col].mean()
            df_cluster_5_ng[col].fillna(mean_value, inplace=True)

    ### SCALING ###

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_cluster_5_ng)
    kmeans = KMeans()
    elbow.fit(scaled_data)#ben ekledim
    elbow.show()

    kmeans = KMeans(n_clusters=3).fit(scaled_data)

    clusters_kmeans = kmeans.labels_

    df_cluster_5["cluster"] = clusters_kmeans

    df_cluster_5["cluster"] = df_cluster_5["cluster"] + 1

    df_cluster_5["cluster"].value_counts()

    st.title(f'Second Model Elbow Method K ={3}')
    st.line_chart(elbow.k_scores_, width=500, use_container_width=True)

    df_cluster_5.to_csv("clustered_final.csv")

    ### CLUSTER VISUALIZATION ###

    df = pd.read_csv('clustered_final.csv')



    # cluster index sabitleme

    cluster_lengths = df.groupby('cluster').size().reset_index(name='cluster_length')

    selected_clusters = cluster_lengths[
        (cluster_lengths['cluster_length'] >= 190) & (cluster_lengths['cluster_length'] <= 200)]

    selected_cluster_numbers = selected_clusters['cluster'].tolist()

    df = df[df['cluster'] == selected_cluster_numbers[0]]


    df_cluster = df["cluster"].value_counts().reset_index()
    df_cluster = df_cluster.sort_values(by="cluster", ascending=False)
    df_cluster.columns = ['cluster', 'Tavsiye']
    #st.dataframe(df_cluster)

    df_cluster_ = df.loc[:, ["cluster", "LATITUDE", "LONGITUDE"]]

    st.title("Demonstration of Map")

    st.map(df_cluster_, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')


   # m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10)

    #st.write(m)

   # for i#ndex, row in df.iterrows():
   #     folium.CircleMarker(
   #         location=[row['LATITUDE'], row['LONGITUDE']],
   #         radius=5,
   #         color='blue',
   #         fill=True,
   #         fill_color='blue'
   #     ).add_to(m)

   # m.save('cluster_unknown.html')





#eski streamlit
    #kmeans = KMeans()
    #elbow = KElbowVisualizer(kmeans, k=(2, 20))
    #elbow.fit(scaled_data)
    #kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(scaled_data)
    #plt.show(block=True)
    #elbow.show()

    #st.write(f'Elbow Method K ={elbow.elbow_value_}')
    #st.line_chart(elbow.k_scores_, width=500, use_container_width=True)

   # clusters_kmeans = kmeans.labels_
   # df["cluster"] = clusters_kmeans
    #df["cluster"] = df["cluster"] + 1
    #df["cluster"].value_counts()
    #df_ = pd.read_csv("traffic_density_202207.csv")

    #unique_coordinates_df_ = df_.drop_duplicates(subset="GEOHASH")

    #unique_geohash = unique_coordinates_df_['GEOHASH'].tolist()
   # unique_latitudes = unique_coordinates_df_['LATITUDE'].tolist()
   # unique_longitudes = unique_coordinates_df_['LONGITUDE'].tolist()

   # df_with_lat_lon = pd.DataFrame(
   #    {'GEOHASH': unique_geohash, 'LATITUDE': unique_latitudes, 'LONGITUDE': unique_longitudes})
   # df = df.merge(df_with_lat_lon, on="GEOHASH")

    df.to_csv("df_son.csv")

   ## df_cluster = df["cluster"].value_counts().reset_index()
   # df_cluster = df_cluster.sort_values(by="cluster", ascending=False)
   # df_cluster.columns = ['cluster', 'Tavsiye']
   # st.dataframe(df_cluster)

    ###################################################
    # Tüm haritaları gösterir.
    ###################################################
  #  for i in range(1, (elbow.elbow_value_ + 1)):
   #     st.write(f'Number Of Cluster ={i} and Count = {df_cluster[df_cluster["cluster"] == i]["Tavsiye"].values[0]}')
    #    df_cluster_ = df[df['cluster'] == i].loc[:, ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
    #    st.map(df_cluster_, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')

###################################################
# Check Box ama bence böyle olmasın her yenilendiğinde cluster değişiyor.
###################################################
#
# Cluster2 = st.checkbox('Cluster_2')
# if Cluster2:
#     st.write(f'Number Of Cluster ={2} and Count = {df_cluster[df_cluster["cluster"] == 2]["Tavsiye"].values[0]}')
#     df_cluster_5 = df[df['cluster'] == 2].loc[:,
#                    ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
#     st.map(df_cluster_5, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')
#
# Cluster3 = st.checkbox('Cluster_3')
# if Cluster3:
#     st.write(f'Number Of Cluster ={3} and Count = {df_cluster[df_cluster["cluster"] == 3]["Tavsiye"].values[0]}')
#     df_cluster_5 = df[df['cluster'] == 3].loc[:,
#                    ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
#     st.map(df_cluster_5, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')
#
# Cluster4 = st.checkbox('Cluster_4')
# if Cluster4:
#     st.write(f'Number Of Cluster ={4} and Count = {df_cluster[df_cluster["cluster"] == 4]["Tavsiye"].values[0]}')
#     df_cluster_5 = df[df['cluster'] == 4].loc[:,
#                    ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
#     st.map(df_cluster_5, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')
#
# Cluster5 = st.checkbox('Cluster_5')
# if Cluster5:
#     st.write(f'Number Of Cluster ={5} and Count = {df_cluster[df_cluster["cluster"] == 5]["Tavsiye"].values[0]}')
#     df_cluster_5 = df[df['cluster'] == 5].loc[:,
#                    ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
#     st.map(df_cluster_5, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')
#
# Cluster6 = st.checkbox('Cluster_6')
# if Cluster6:
#     st.write(f'Number Of Cluster ={6} and Count = {df_cluster[df_cluster["cluster"] == 6]["Tavsiye"].values[0]}')
#     df_cluster_5 = df[df['cluster'] == 6].loc[:,
#                    ["cluster", "LATITUDE_x", "LONGITUDE_x"]]
#     st.map(df_cluster_5, latitude='LATITUDE_x', longitude='LONGITUDE_x', size=60, color='#0044ff')

elif page == 'Finding a shortest route':

    def user_input_features():
        # DataSize = st.sidebar.slider('DataSize', 100, 450, 1000)
        # NumberOfCluster = st.sidebar.slider('NumberOfCluster', min_value=1, max_value=18, step=1)
        # Varience = st.sidebar.slider('Varience', 1, 5, 10)
        # NumberofCentroids = st.sidebar.slider('NumberofCentroids', 1, 5, 10)
        LATITUDE = st.sidebar.number_input('LATITUDE')
        LONGITUDE = st.sidebar.number_input('LONGITUDE')
        data_1 = {'LATITUDE': LATITUDE,
                'LONGITUDE': LONGITUDE}

        features = pd.DataFrame(data_1, index=[0])
        return features


    input_df = user_input_features()
    st.dataframe(input_df)

    if st.button('Find shortest route'):
        # Kullanıcının girdiği konum

        st.title("Your Location")
        st.map(input_df, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')
        df = pd.read_csv("df_son.csv")

        st.title("Finding a shortest route")

        df_locaiton = df.loc[:, ["GEOHASH", "LATITUDE", "LONGITUDE", "cluster"]]#x çıktı
        df_locaiton["LATITUDE_input"] = input_df["LATITUDE"]
        df_locaiton = df_locaiton.fillna(input_df["LATITUDE"].values[0])
        df_locaiton["LONGITUDE_input"] = input_df["LONGITUDE"]
        df_locaiton = df_locaiton.fillna(input_df["LONGITUDE"].values[0])


        def haversine(lat1, lon1, lat2, lon2):
            # Dünya'nın yarı çapı
            R = 6371  # Yarı çapı km olarak alabilirsiniz.

            # Lat ve Lon'u radyanlara dönüştürün
            lat1 = math.radians(lat1)
            lon1 = math.radians(lon1)
            lat2 = math.radians(lat2)
            lon2 = math.radians(lon2)

            # Haversine formülü
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            return distance


        # Mesafeyi hesaplayın ve yeni bir sütun ekleyin
        df_locaiton['Mesafe'] = df_locaiton.apply(
            lambda row: haversine(row['LATITUDE_input'], row['LONGITUDE_input'], row['LATITUDE'], row['LONGITUDE']),
            axis=1)
        df_kısa = df_locaiton.sort_values(by='Mesafe', ascending=True).head()
        st.map(df_kısa, latitude='LATITUDE', longitude='LONGITUDE', size=60, color='#0044ff')


else:
    st.title("Contact")

    col1, col2, = st.columns(2)

    with col1:
        st.title("Miuul")
        image_url = "https://miuul.com/image/theme/logo-dark.png"
        st.image(image_url, use_column_width=True)

    with col2:
        st.title("Veri Bilimi Okulu")
        image_url = "https://www.veribilimiokulu.com/wp-content/uploads/2020/12/veribilimiokulu_logo-crop.png"
        st.image(image_url, use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        link = "[Miuul](https://miuul.com)"
        st.markdown(link, unsafe_allow_html=True)

    with col2:
        link = "[Veri Bilimi Okulu](https://bootcamp.veribilimiokulu.com/bootcamp-programlari/veri-bilimci-yetistirme-programi/)"
        st.markdown(link, unsafe_allow_html=True)

    col1, = st.columns(1)

    with col1:
        st.title("Kaynakça 🎤")
        video_url = "https://www.youtube.com/watch?v=Ww9M0WJfGN8"
        st.video(video_url)

    col3, = st.columns(1)
    with col3:
        st.title("Linkedin")
        st.write("[Yasemin Ergün](https://www.linkedin.com/in/yaseminergun)")
        st.write("[Fatma Yağmurlu](http://www.linkedin.com/in/fatmayagmurlu)")
        st.write("[Oğuzcan Maden](https://www.linkedin.com/in/oguzcnmdn/)")
        st.write("[Cenk Bayender](https://www.linkedin.com/in/cenk-bayender-98a285130)")
        st.write("[Ozan Bahar](https://www.linkedin.com/in/ozan-bahar/)")
