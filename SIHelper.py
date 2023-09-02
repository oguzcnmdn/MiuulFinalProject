import pandas as pd
class Summarizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe['DATE_TIME'] = pd.to_datetime(self.dataframe['DATE_TIME'])
        self.custom_month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November',
                                   'December']

    def find_hourly_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH']).agg(
            AVG_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_HOURLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_daily_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'DAY_N']).agg(
            AVG_NUM_VEHICLES_DAILY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_DAILY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_daily_statistics_C(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'DAY_NUMBER']).agg(
            AVG_NUM_VEHICLES_DAILY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_DAILY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        result_df.sort_values(by='DAY_NUMBER')
        avg_day_df = result_df.pivot(index='GEOHASH', columns='DAY_NUMBER',
                                     values='AVG_NUM_VEHICLES_DAILY').reset_index()
        total_day_df = result_df.pivot(index='GEOHASH', columns='DAY_NUMBER',
                                       values='TOTAL_NUM_VEHICLES_DAILY').reset_index()
        avg_day_df.columns = [f'AVG_NUM_VEHICLES_DAY_{col}' if col != 'GEOHASH'
                              else col for col in avg_day_df.columns]
        total_day_df.columns = [f'TOTAL_NUM_VEHICLES_DAY_{col}' if col != 'GEOHASH'
                                else col for col in total_day_df.columns]
        result_df = avg_day_df.merge(total_day_df, on=['GEOHASH'])
        return result_df

    def find_weekly_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'WEEK']).agg(
            AVG_NUM_VEHICLES_WEEKLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_WEEKLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_weekly_statistics_C(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'WEEK']).agg(
            AVG_NUM_VEHICLES_WEEKLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_WEEKLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        result_df.sort_values(by='WEEK')
        avg_week_df = result_df.pivot(index='GEOHASH', columns='WEEK',
                                      values='AVG_NUM_VEHICLES_WEEKLY').reset_index()
        total_week_df = result_df.pivot(index='GEOHASH', columns='WEEK',
                                        values='TOTAL_NUM_VEHICLES_WEEKLY').reset_index()
        avg_week_df.columns = [f'AVG_NUM_VEHICLES_WEEK_{col}' if col != 'GEOHASH'
                               else col for col in avg_week_df.columns]
        total_week_df.columns = [f'TOTAL_NUM_VEHICLES_WEEK_{col}' if col != 'GEOHASH'
                                 else col for col in total_week_df.columns]
        result_df = avg_week_df.merge(total_week_df, on=['GEOHASH'])
        return result_df

    def find_monthly_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'MONTH']).agg(
            AVG_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def find_monthly_statistics_C(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'MONTH']).agg(
            AVG_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_MONTHLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        result_df['MONTH'] = pd.Categorical(result_df['MONTH'], categories=self.custom_month_order, ordered=True)
        result_df.sort_values(by='MONTH')
        avg_month_df = result_df.pivot(index='GEOHASH', columns='MONTH',
                                       values='AVG_NUM_VEHICLES_MONTHLY').reset_index()
        total_month_df = result_df.pivot(index='GEOHASH', columns='MONTH',
                                         values='TOTAL_NUM_VEHICLES_MONTHLY').reset_index()
        avg_month_df.columns = [f'AVG_NUM_VEHICLES_MONTH_{col.capitalize()}' if col != 'GEOHASH'
                                else col for col in avg_month_df.columns]
        total_month_df.columns = [f'TOTAL_NUM_VEHICLES_MONTH_{col.capitalize()}' if col != 'GEOHASH'
                                  else col for col in total_month_df.columns]
        result_df = avg_month_df.merge(total_month_df, on=['GEOHASH'])
        return result_df

    def find_yearly_statistics(self):
        result_df = self.dataframe.groupby(['GEOHASH', 'YEAR']).agg(
            AVG_NUM_VEHICLES_YEARLY=('NUMBER_OF_VEHICLES', 'mean'),
            TOTAL_NUM_VEHICLES_YEARLY=('NUMBER_OF_VEHICLES', 'sum')).reset_index()
        return result_df

    def create_summary_dataframe(self):
        hourly_df = self.find_hourly_statistics()
        monthly_df = self.find_monthly_statistics()
        yearly_df = self.find_yearly_statistics()
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
        yearly_df = self.find_yearly_statistics()
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