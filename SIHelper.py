import pandas as pd
class Summarizer:

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

        This constructor initializes the Summarizer object with the provided DataFrame. It also converts the 'DATE_TIME' column
        of the DataFrame to datetime format and sets up custom ordering for months and seasons.
        """

        self.dataframe = dataframe
        self.dataframe['DATE_TIME'] = pd.to_datetime(self.dataframe['DATE_TIME'])
        self.custom_month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November',
                                   'December']
        self.custom_season_order = ['Fall', 'Winter', 'Spring', 'Summer']

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
            TOTAL_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
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
            A DataFrame containing custom statistics for each unique combination of 'GEOHASH' and the specified group_column, including:
            - AVG_NUM_VEHICLES: Mean number of vehicles for each combination.
            - TOTAL_NUM_VEHICLES: Total number of vehicles for each combination.
        """

        result_df = self.dataframe.groupby(['GEOHASH', group_column]).agg(
            AVG_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
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
            TOTAL_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        if group_column == 'MONTH':
            result_df[group_column] = pd.Categorical(result_df[group_column], categories=self.custom_month_order,
                                                     ordered=True)
            result_df.sort_values(by=group_column)
        elif group_column == 'SEASON':
            result_df[group_column] = pd.Categorical(result_df[group_column], categories=self.custom_season_order,
                                                     ordered=True)
            result_df.sort_values(by=group_column)
        else:
            result_df.sort_values(by=group_column)
        avg_df = result_df.pivot(index='GEOHASH', columns=group_column,
                                       values='AVG_NUM_VEHICLES').reset_index()
        total_df = result_df.pivot(index='GEOHASH', columns=group_column,
                                         values='TOTAL_NUM_VEHICLES').reset_index()
        avg_df.columns = [f'AVG_NUM_VEHICLES_{group_column}_{col}' if col != 'GEOHASH'
                                else col for col in avg_df.columns]
        total_df.columns = [f'TOTAL_NUM_VEHICLES_{group_column}_{col}' if col != 'GEOHASH'
                                  else col for col in total_df.columns]
        result_df = avg_df.merge(total_df, on=['GEOHASH'])
        return result_df


