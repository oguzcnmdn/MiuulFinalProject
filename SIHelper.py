import math
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import geohash2

class Abraham:
    """
    Class for processing and adding features to geospatial and temporal data.

    Args:
        interval (int): Time interval for distance calculations (default is 20).

    Attributes:
        interval (int): Time interval for distance calculations.
        time_labels_N (list): Labels for time intervals (Daytime).
        time_bins_N (list): Time bins for Daytime intervals.
        time_labels_D (list): Labels for time intervals (Detailed).
        time_bins_D (list): Time bins for Detailed time intervals.
        school_break_periods (list): List of tuples representing school break periods.
        public_holidays (list): List of public holidays.

    Methods:
        create_unique_coordinates_dataframe(dataframe):
            Creates a DataFrame with unique geospatial coordinates.

        calculate_distance(start_lat, start_lon, end_lat, end_lon):
            Calculates the distance between two points on the Earth's surface.

        add_distance_features(dataframe):
            Adds distance-related features to the DataFrame.

        add_time_features(dataframe):
            Adds temporal features to the DataFrame.

    """

    def __init__(self, interval=20):
        """
        Initializes an instance of the Abraham class.

        Args:
            interval (int, optional): Time interval for distance calculations (default is 20).

        Attributes:
            interval (int): Time interval for distance calculations.
            time_labels_N (list): Labels for Daytime intervals.
            time_bins_N (list): Time bins for Daytime intervals.
            time_labels_D (list): Labels for Detailed time intervals.
            time_bins_D (list): Time bins for Detailed time intervals.
            school_break_periods (list): List of tuples representing school break periods.
            public_holidays (list): List of public holidays.

        """
        self.interval = interval
        self.time_labels_N = ['Late Night', 'Morning', 'Late Morning', 'Afternoon', 'Late Afternoon', 'Evening',
                              'Night']
        self.time_bins_N = [0, 5, 9, 12, 15, 18, 21, 24]

        self.time_labels_D = ['Midnight 24:00-01:59', 'Early Morning 02:00-04:59','Morning 05:00-06:59',
                             'Morning Rush Hour 07:00-09:59','Late Morning 10:00-12:59',
                             'Afternoon 13:00-14:59', 'Late Afternoon 15:00-16:59',
                             'Evening Rush Hour 17:00-19:59', 'Night 20:00-23:59']

        self.time_bins_D = [0, 2, 5, 7, 10, 13, 15, 17, 20, 24]

        self.school_break_periods = [
            ('2022-07-01', '2022-09-11', 'Summer Break'),
            ('2022-11-14', '2022-11-18', 'Partial Midterm Break'),
            ('2023-01-23', '2023-02-03', 'Midterm Break'),
            ('2023-04-17', '2023-04-20', 'Partial Midterm Break'),
            ('2023-06-17', '2023-06-30', 'Summer Break')
        ]

        self.public_holidays = [
            '2022-07-08', '2022-07-09', '2022-07-10', '2022-07-11', '2022-07-12', '2022-07-15',
            '2022-08-30', '2022-10-28', '2022-10-29', '2023-01-01', '2023-04-20', '2023-04-21',
            '2023-04-22', '2023-04-23', '2023-05-01', '2023-05-19', '2023-06-27', '2023-06-28',
            '2023-06-29', '2023-06-30', '2023-07-01', '2023-07-15'
        ]

    def create_unique_coordinates_dataframe(self, dataframe):
        """
        Creates a DataFrame with unique geospatial coordinates.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with geospatial information.

        Returns:
            pd.DataFrame: A DataFrame with unique geospatial coordinates.

        """
        unique_coordinates_dataframe = dataframe.drop_duplicates(subset='GEOHASH')

        unique_geohash = unique_coordinates_dataframe['GEOHASH'].tolist()
        unique_latitudes = unique_coordinates_dataframe['LATITUDE'].tolist()
        unique_longitudes = unique_coordinates_dataframe['LONGITUDE'].tolist()

        dataframe = pd.DataFrame({'GEOHASH': unique_geohash, 'LATITUDE': unique_latitudes, 'LONGITUDE': unique_longitudes})

        return dataframe

    def calculate_distance(self, start_lat, start_lon, end_lat, end_lon):
        """
        Calculates the distance between two points on the Earth's surface using the Haversine formula.

        Args:
            start_lat (float): Latitude of the starting point in degrees.
            start_lon (float): Longitude of the starting point in degrees.
            end_lat (float): Latitude of the ending point in degrees.
            end_lon (float): Longitude of the ending point in degrees.

        Returns:
            float: The calculated distance in kilometers.

        Constants:
            R (float): Radius of the Earth in kilometers.

        Formula:
            The Haversine formula is used to calculate the distance:

            delta_latitude = math.radians(end_lat - start_lat)
            delta_longitude = math.radians(end_lon - start_lon)

            a = math.sin(delta_latitude / 2) ** 2 +
                math.cos(math.radians(start_lat)) * math.cos(math.radians(end_lat)) *
                math.sin(delta_longitude / 2) ** 2

            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance = R * c

        Note:
            - The radius of the Earth (R) is assumed to be approximately 6371 kilometers.

        """
        R = 6371
        delta_latitude = math.radians(end_lat - start_lat)
        delta_longitude = math.radians(end_lon - start_lon)
        a = math.sin(delta_latitude / 2) * math.sin(delta_latitude / 2) + math.cos(math.radians(start_lat)) * math.cos(
            math.radians(end_lat)) * math.sin(delta_longitude / 2) * math.sin(delta_longitude / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance


    def add_distance_features(self, dataframe):
        """
        Adds distance-related features to the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with geospatial information.

        Returns:
            pd.DataFrame: DataFrame with added distance features.

        Calculated Features:
            - 'START_LAT': Latitude of the starting point.
            - 'START_LON': Longitude of the starting point.
            - 'END_LAT': Latitude of the ending point (shifted).
            - 'END_LON': Longitude of the ending point (shifted).
            - 'DISTANCE_KM': Distance in kilometers between starting and ending points.
            - 'DISTANCE_INTERVAL': Distance rounded to the nearest interval.
            - 'TRAFFIC_DENSITY': Traffic density based on the number of vehicles, distance, and average speed.

        """
        dataframe['START_LAT'] = dataframe['LATITUDE']
        dataframe['START_LON'] = dataframe['LONGITUDE']
        dataframe['END_LAT'] = dataframe['LATITUDE'].shift(-1)
        dataframe['END_LON'] = dataframe['LONGITUDE'].shift(-1)

        dataframe['DISTANCE_KM'] = dataframe.apply(
            lambda row: self.calculate_distance(row['START_LAT'], row['START_LON'], row['END_LAT'], row['END_LON']),
            axis=1)

        dataframe['DISTANCE_INTERVAL'] = (dataframe['DISTANCE_KM'] // self.interval) * self.interval

        dataframe['TRAFFIC_DENSITY'] = (dataframe["NUMBER_OF_VEHICLES"] / dataframe["DISTANCE_KM"]) *\
                                       dataframe["AVERAGE_SPEED"]

        return dataframe

    def add_time_features(self, dataframe):
        """
        Adds temporal features to the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with temporal information.

        Returns:
            pd.DataFrame: DataFrame with added temporal features.

        Calculated Features:
            - 'DATE_TIME': Conversion of the 'DATE_TIME' column to datetime format.
            - 'HOUR_N': Extracted hour from the datetime (24-hour format).
            - 'HOUR_D': Detailed time information based on the datetime.
            - 'DAY': Name of the day.
            - 'DAY_NUMBER': Day of the month.
            - 'WEEK': Week number.
            - 'MONTH': Month name.
            - 'YEAR': Year.
            - 'SEASON': Season based on the month.
            - 'DAY_INTERVAL': Weekdays or Weekends.
            - 'TIME_INTERVAL_N': Categorized time interval (Daytime).
            - 'TIME_INTERVAL_D': Detailed categorized time interval.
            - 'BREAK_PERIOD': School break period.
            - 'PUBLIC_HOLIDAY': Whether it's a public holiday or not.

        """
        dataframe['DATE_TIME'] = pd.to_datetime(dataframe['DATE_TIME'])

        dataframe['HOUR_N'] = dataframe['DATE_TIME'].dt.hour

        def date_month_year_day_names_time(date):
            year = date.year
            month_number = date.month
            week_number = date.week
            day_number = date.day
            time = date.strftime('%H:%M:%S')

            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            month_name = month_names[month_number - 1]

            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = day_names[date.weekday()]

            return time, day_name, day_number, week_number, month_name, year

        dataframe['HOUR_D'], dataframe['DAY'], dataframe['DAY_NUMBER'], dataframe['WEEK'], dataframe['MONTH'],\
            dataframe['YEAR'] = zip(*dataframe['DATE_TIME'].apply(date_month_year_day_names_time))

        dataframe['SEASON'] = dataframe['MONTH'].apply(lambda x: 'Winter' if x in ['December', 'January', 'February']
                                                        else 'Spring' if x in ['March', 'April', 'May']
                                                        else 'Summer' if x in ['June', 'July', 'August']
                                                        else 'Fall')

        dataframe['DAY_INTERVAL'] = dataframe['DAY'].apply(lambda x: 'Weekends' if x in ['Saturday', 'Sunday']
        else 'Weekdays')

        dataframe['TIME_INTERVAL_N'] = pd.cut(dataframe['HOUR_N'], bins=self.time_bins_N, labels=self.time_labels_N,
                                              include_lowest=True)
        dataframe['TIME_INTERVAL_D'] = pd.cut(dataframe['HOUR_N'], bins=self.time_bins_D, labels=self.time_labels_D,
                                              include_lowest=True)

        dataframe['BREAK_PERIOD'] = 'None'
        for start_date, end_date, break_type in self.school_break_periods:
            mask = (dataframe['DATE_TIME'] >= pd.to_datetime(start_date)) & (dataframe['DATE_TIME'] <= pd.to_datetime(end_date))
            dataframe.loc[mask, 'BREAK_PERIOD'] = break_type

        dataframe['PUBLIC_HOLIDAY'] = 'No'
        for public_holiday in self.public_holidays:
            mask = (dataframe['DATE_TIME'].dt.date == pd.to_datetime(public_holiday).date())
            dataframe.loc[mask, 'PUBLIC_HOLIDAY'] = 'Yes'

        return dataframe

class Jerzy:

    """
    A class for summarizing and aggregating data in a Pandas DataFrame.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The input DataFrame containing data to be summarized.

    Attributes:
    -----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data.

    custom_month_order : list of str
        A custom order for months when grouping data by 'MONTH'.

    custom_season_order : list of str
        A custom order for seasons when grouping data by 'SEASON'.

    Methods:
    -----------
    find_based_statistics(self):
        Calculate and return basic statistics based on the 'GEOHASH' column, including:
        - AVG_NUM_VEHICLES_HOURLY: Mean number of vehicles per hour per geohash.
        - TOTAL_NUM_VEHICLES_HOURLY: Total number of vehicles per hour per geohash.

    find_custom_statistics(self, group_column):
        Calculate and return custom statistics based on the 'GEOHASH' and specified group_column.
        - group_column (str): The column by which to group the data.

    find_custom_statistics_C(self, group_column):
        Calculate and return custom statistics with ordered categories based on the 'GEOHASH'
        and specified group_column.
        - group_column (str): The column by which to group the data.
    """

    def __init__(self, dataframe):

        """
        Initialize a Summarizer object.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input DataFrame containing data to be summarized.

        Attributes:
        -----------
        dataframe : pandas.DataFrame
            The input DataFrame containing the data.

        custom_month_order : list of str
            A custom order for months when grouping data by 'MONTH'.

        custom_season_order : list of str
            A custom order for seasons when grouping data by 'SEASON'.

        This constructor initializes the Summarizer object with the provided DataFrame. It also converts the 'DATE_TIME'
        column of the DataFrame to datetime format and sets up custom ordering for months and seasons.
        """

        self.dataframe = dataframe
        self.dataframe['DATE_TIME'] = pd.to_datetime(self.dataframe['DATE_TIME'])
        self.custom_month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November',
                                   'December']
        self.custom_season_order = ['Fall', 'Winter', 'Spring', 'Summer']
        self.custom_day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    def find_based_statistics(self):

        """
        Calculate and return basic statistics based on the 'GEOHASH' column.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing basic statistics for each unique 'GEOHASH', including:
            - AVG_NUM_VEHICLES_HOURLY: Mean number of vehicles per hour per geohash.
            - TOTAL_NUM_VEHICLES_HOURLY: Total number of vehicles per hour per geohash.
        """

        result_df = self.dataframe.groupby(['GEOHASH']).agg(
            AVG_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'sum'),
            AVG_TRAFFIC_DENSITY_HOURLY=('TRAFFIC_DENSITY', 'mean')).reset_index()
        return result_df

    def find_custom_statistics(self, group_column):

        """
        Calculate and return custom statistics based on the 'GEOHASH' and specified group_column.

        Parameters:
        -----------
        group_column : str
            The column by which to group the data.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing custom statistics for each unique combination of 'GEOHASH' and the specified
            group_column, including:
            - AVG_NUM_VEHICLES: Mean number of vehicles for each combination.
            - TOTAL_NUM_VEHICLES: Total number of vehicles for each combination.
        """

        result_df = self.dataframe.groupby(['GEOHASH', group_column]).agg(
            AVG_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'sum'),
            AVG_TRAFFIC_DENSITY=('TRAFFIC_DENSITY', 'mean')).reset_index()
        result_df.columns = ['GEOHASH', f'{group_column}', f'AVG_NUM_VEHICLES_{group_column}',
                            f'TOTAL_NUM_VEHICLES_{group_column}', f'AVG_TRAFFIC_DENSITY_{group_column}']
        return result_df

    def find_custom_statistics_C(self, group_column):

        """
         Calculate and return custom statistics with ordered categories based on the 'GEOHASH'
         and specified group_column.

         Parameters:
         -----------
         group_column : str
             The column by which to group the data.

         Returns:
         --------
         pandas.DataFrame
             A DataFrame containing custom statistics with ordered categories based on 'GEOHASH'
             and the specified group_column. The ordering is applied to 'SEASON' and 'MONTH'.
         """

        result_df = self.dataframe.groupby(['GEOHASH', group_column]).agg(
            AVG_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'sum'),
            AVG_TRAFFIC_DENSITY=('TRAFFIC_DENSITY', 'mean')).reset_index()
        if group_column == 'MONTH':
            result_df[group_column] = pd.Categorical(result_df[group_column], categories=self.custom_month_order,
                                                     ordered=True)
            result_df.sort_values(by=group_column)
        elif group_column == 'SEASON':
            result_df[group_column] = pd.Categorical(result_df[group_column], categories=self.custom_season_order,
                                                     ordered=True)
            result_df.sort_values(by=group_column)
        elif group_column == 'DAY':
            result_df[group_column] = pd.Categorical(result_df[group_column], categories=self.custom_day_order,
                                                     ordered=True)
            result_df.sort_values(by=group_column)
        else:
            result_df.sort_values(by=group_column)
        avg_df = result_df.pivot(index='GEOHASH', columns=group_column,
                                       values='AVG_NUM_VEHICLES').reset_index()
        total_df = result_df.pivot(index='GEOHASH', columns=group_column,
                                         values='TOTAL_NUM_VEHICLES').reset_index()
        density_df = result_df.pivot(index='GEOHASH', columns=group_column,
                                         values='AVG_TRAFFIC_DENSITY').reset_index()
        avg_df.columns = [f'AVG_NUM_VEHICLES_{group_column}_{col}' if col != 'GEOHASH'
                                else col for col in avg_df.columns]
        total_df.columns = [f'TOTAL_NUM_VEHICLES_{group_column}_{col}' if col != 'GEOHASH'
                                  else col for col in total_df.columns]
        density_df.columns = [f'AVG_TRAFFIC_DENSITY{group_column}_{col}' if col != 'GEOHASH'
                                  else col for col in total_df.columns]
        result_df = avg_df.merge(total_df, on=['GEOHASH']).merge(density_df, on=['GEOHASH'])
        return result_df

class Marco:
    """
    Class for processing geospatial data.

    Methods:
        find_lon_lan_dataframe(dataframe):
            Finds geohash values for latitude and longitude coordinates.

        haversine(latitude1, longitude1, latitude2, longitude2):
            Calculates the distance between two points on the Earth's surface using the Haversine formula.

        count_places_near_traffic(place_dataframe, final_dataframe):
            Counts the number of places near traffic points within a specified radius.

    """
    def find_lon_lan_dataframe(self, dataframe):
        """
        Finds geohash values for latitude and longitude coordinates in the DataFrame.

        Args:
            dataframe (pd.DataFrame): Input DataFrame with 'LATITUDE' and 'LONGITUDE' columns.

        Returns:
            pd.DataFrame: DataFrame with an added 'GEOHASH' column containing geohash values.

        """
        latitude = dataframe["LATITUDE"]
        longitude = dataframe["LONGITUDE"]

        dataframe["LATITUDE"] = dataframe["LATITUDE"].astype(float)
        dataframe["LONGITUDE"] = dataframe["LONGITUDE"].astype(float)

        dataframe["GEOHASH"] = ""
        for index, row in dataframe.iterrows():
            latitude = row["LATITUDE"]
            longitude = row["LONGITUDE"]
            geohash_code = geohash2.encode(latitude, longitude)
            dataframe.at[index, "GEOHASH"] = geohash_code

        return dataframe

    def haversine(self, latitude1, longitude1, latitude2, longitude2):
        """
        Calculates the distance between two points on the Earth's surface using the Haversine formula.

        Args:
            latitude1 (float): Latitude of the first point.
            longitude1 (float): Longitude of the first point.
            latitude2 (float): Latitude of the second point.
            longitude2 (float): Longitude of the second point.

        Returns:
            float: The calculated distance in kilometers.

        Constants:
            R (float): Radius of the Earth in kilometers.

        Formula:
            The Haversine formula is used to calculate the distance:

            dlat = latitude2 - latitude1
            dlon = longitude2 - longitude1

            a = sin(dlat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c

        Note:
            - The radius of the Earth (R) is assumed to be approximately 6371 kilometers.

        """
        R = 6371

        latitude1, longitude1, latitude2, longitude2 = map(float, [latitude1, longitude1, latitude2, longitude2])

        latitude1, longitude1, latitude2, longitude2 = map(radians, [latitude1, longitude1, latitude2, longitude2])

        dlat = latitude2 - latitude1
        dlon = longitude2 - longitude1

        a = sin(dlat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def count_places_near_traffic(self, place_dataframe, final_dataframe):
        """
        Counts the number of places near traffic points within a specified radius.

        Args:
            place_dataframe (pd.DataFrame): DataFrame containing geohash values for places.
            final_dataframe (pd.DataFrame): DataFrame containing geohash values for traffic points.

        Returns:
            dict: A dictionary mapping traffic geohash codes to the count of nearby places.

        """

        place_g_list = place_dataframe["GEOHASH"].tolist()
        final_g_list = final_dataframe["GEOHASH"].tolist()

        result = {}

        for i in final_g_list:
            count = 0
            latitude1, longitude1 = geohash2.decode(i)

            for place in place_g_list:
                latitude2, longitude2 = geohash2.decode(place)

                if self.haversine(latitude1, longitude1, latitude2, longitude2) <= 20:
                    count += 1

            result[i] = count

        return result

