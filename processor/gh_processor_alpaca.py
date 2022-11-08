from __future__ import annotations

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
from stockstats import StockDataFrame as Sdf


class AlpacaProcessor:
    def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, api=None):
        if api is None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def download_data(
        self, ticker_list, start_date, end_date, time_interval, adjst
    ) -> pd.DataFrame:
        """
        ticker_list : list string of ticket
        time_interval: time interval
        start_date : start date of America/New_York time
        end_date : end date of America/New_York time
        The function tries to retrieve the data, between the start date and the end date, from the Alpaca server.
        if time_interval < 1D: period of data retrieved is the opening time of the New York Stock Exchange (NYSE) (from 9:30 am to 4:00 pm), in UTC offset zone.
        if time_interval >= 1D: each bar is the midnight of the day in America/New_York time, in UTC offset zone.
        """
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # download
        NY = "America/New_York"
        start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
        end_date = pd.Timestamp(end_date + " 16:00:00", tz=NY)

        data = pd.DataFrame()
        for stock in ticker_list:
            flg = True
            start = start_date
            while flg:
                bars = self.processor.api.get_bars(stock, time_interval, start.isoformat(), end_date.isoformat(), adjustment = adjst, limit=100000).df
                bars["tic"] = stock
                data = pd.concat([data, bars])
                if len(bars) == 100000:
                    start = bars.index[-1]
                else:
                    flg = False
        data = data.drop_duplicates()

        # filter opening time of the New York Stock Exchange (NYSE) (from 9:30 am to 4:00 pm) if time_interval < 1D
        if pd.Timedelta(time_interval).delta < pd.Timedelta("1D").delta:
            NYSE_open_hour = "14:30"  # in UTC
            NYSE_close_hour = "21:00"  # in UTC
            data_df = data.between_time(NYSE_open_hour, NYSE_close_hour)
        else:
            data_df = data

        # reformat to finrl expected schema
        data_df = data_df.reset_index()
        data_df["timestamp"] = data_df["timestamp"].apply(lambda x: x.tz_convert(NY))
        print("Data successfully dowloaded")
        return data_df

    def clean_data(self, df):
        #tic_list = np.unique(df.tic.values)
        tic_list = df["tic"].unique()
        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index
        times = []
        t_steps = int((pd.Timedelta("6.5H").delta / pd.Timedelta(self.time_interval).delta))
        for day in trading_days:
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize("America/New_York")

            for i in range(t_steps):
                times.append(current_time)
                current_time += pd.Timedelta(self.time_interval)

        new_df = pd.DataFrame()
        for tic in tic_list:
            tic_df = df[df.tic == tic]
            tic_df = tic_df.set_index("timestamp").reindex(times, fill_value=np.nan)

            if np.isnan(tic_df.iloc[0]["close"]):
                print("The price of the first row for ticker ",tic," is NaN. ","It will filled with the first valid price.")
                try:
                    tic_df.iloc[[0]] = tic_df.dropna(subset="close").iloc[0,]
                except:
                    raise Exception("All data is missing for ticker ", tic)
            tic_df = tic_df.fillna(method="ffill")
            
            new_df = pd.concat([new_df, tic_df])

        new_df = new_df.reset_index()
        print("Data clean finished!")

        return new_df

    def add_technical_indicator(
        self,
        df,
        tech_indicator_list=[
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    ):
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        #for ticker in unique_ticker:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]][["tic", "timestamp"] + tech_indicator_list]
            indicator_df = pd.concat([indicator_df, temp_indicator])

        df = df.merge(indicator_df, on=["tic", "timestamp"], how="left")
        df = df.sort_values(by=["timestamp", "tic"])

        print("Succesfully add technical indicators")
        return df

    def add_vix(self, data, adjst):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval, adjst)
        cleaned_vix = self.clean_data(vix_df)
        vix = cleaned_vix[["timestamp", "close"]]
        vix = vix.rename(columns={"close": "VIXY"})

        df = data.copy()
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df["tic"].unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(start, end)
        trading_days = df.strftime("%Y-%m-%d").to_list()

        return trading_days

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_bars([tic], time_interval, limit=limit).df  # [tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = data_df.append(barset)

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_bars(["VIXY"], time_interval, limit=1).df
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb