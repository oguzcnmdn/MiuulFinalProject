import pandas as pd
class Summarizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe['DATE_TIME'] = pd.to_datetime(self.dataframe['DATE_TIME'])
        self.custom_month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November',
                                   'December']
        self.custom_season_order = ['Autumn', 'Winter', 'Spring', 'Summer']

    def find_based_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH']).agg(
            AVG_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_custom_statistics(self, group_column):
        result_df = self.dataframe.groupby(['GEOHASH', group_column]).agg(
            AVG_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_custom_statistics_C(self, group_column):
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

    def create_summary_dataframe(self):
        hourly_df = self.find_hourly_statistics()
        monthly_df = self.find_monthly_statistics()
        yearly_df = self.find_annual_statistics()
        daily_df = self.find_daily_statistics()
        weekly_df = self.find_weekly_statistics()

        # Merge hourly, monthly, yearly, daily, and weekly data
        merged_df = (
            hourly_df
            .merge(monthly_df, on=['GEOHASH'])
            .merge(yearly_df, on=['GEOHASH'])
            .merge(daily_df, on=['GEOHASH'])
            .merge(weekly_df, on=['GEOHASH'])
        )

    def create_summary_dataframe_C(self):
        hourly_df = self.find_hourly_statistics()
        monthly_df = self.find_monthly_statistics()
        yearly_df = self.find_annual_statistics()
        daily_df = self.find_daily_statistics()
        weekly_df = self.find_weekly_statistics()
        # Merge hourly, monthly, yearly, daily, and weekly data
        merged_df = (
            hourly_df
            .merge(monthly_df, on=['GEOHASH'])
            .merge(yearly_df, on=['GEOHASH'])
            .merge(daily_df, on=['GEOHASH'])
            .merge(weekly_df, on=['GEOHASH'])
        )

        return merged_df