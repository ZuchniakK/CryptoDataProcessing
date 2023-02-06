import copy
import io
import json
import pickle
import random
import zipfile
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from functools import partial
from math import gcd
from os import makedirs
from os.path import exists, join
from time import monotonic

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import seaborn as sns
import talib
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler


def download_extract_zip(url):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield zipinfo.filename, thefile


def get_trades(
    symbol,
    year,
    month,
    day,
    save_trades=True,
    binance_base_url="https://data.binance.vision",
    trades_dir="data/trades",
):
    filename = join(
        trades_dir, symbol.upper(), f"{symbol.upper()}-trades-{year}-{month}-{day}.csv"
    )
    file_exists = exists(filename)
    if file_exists:
        df = pd.read_csv(filename)
        df.index = pd.to_datetime(df.timestamp, unit="ms")
        # df.index = pd.to_datetime(df.index)
        return df

    url = f"{binance_base_url}/data/spot/daily/trades/{symbol.upper()}/{symbol.upper()}-trades-{year}-{month}-{day}.zip"
    file_gen = download_extract_zip(url)
    df = pd.read_csv(
        next(file_gen)[1],
        usecols=[0, 1, 2, 3, 4, 5],
        names=["tradeId", "price", "qty", "quoteQty", "timestamp", "isBuyerMaker"],
    )

    # we add one line at the beginning and end of the dataframe, with the first possible and last possible timestamp
    # of a given day, with the copied values from the first and last trading event. This way we avoid missing time slots
    ms_in_day = 1000 * 3600 * 24
    first_day_timestamp = df["timestamp"][0] - df["timestamp"][0] % ms_in_day
    last_day_timestamp = first_day_timestamp + ms_in_day - 1
    first_indices = df.iloc[0].values
    first_indices[2] = 0
    first_indices[3] = 0
    first_indices[4] = first_day_timestamp
    last_indices = df.iloc[-1].values
    last_indices[2] = 0
    last_indices[3] = 0
    last_indices[4] = last_day_timestamp
    df = pd.concat(
        [
            pd.DataFrame(
                [first_indices],
                columns=[
                    "tradeId",
                    "price",
                    "qty",
                    "quoteQty",
                    "timestamp",
                    "isBuyerMaker",
                ],
            ),
            df,
            pd.DataFrame(
                [last_indices],
                columns=[
                    "tradeId",
                    "price",
                    "qty",
                    "quoteQty",
                    "timestamp",
                    "isBuyerMaker",
                ],
            ),
        ],
        ignore_index=True,
    )

    # Converting the index as date
    df.index = pd.to_datetime(df.timestamp, unit="ms")
    # df.pop('timestamp')
    if save_trades:
        makedirs(join(trades_dir, symbol.upper()), exist_ok=True)
        df.to_csv(filename, index=False)
    return df


def average(values, weights=None, axis=None):
    try:
        return np.average(values, weights=weights, axis=axis)
    except ZeroDivisionError:
        if values.shape[0] == 1:
            return values[0]
        else:
            return np.nan


def trades2ohlc(
    df_trades,
    resampling_frequency=1,
    offset=0,
    base_time_offset=pd.tseries.offsets.Minute(),
    fill_column="close",
):
    resampling_frequency = resampling_frequency * base_time_offset
    offset = offset * base_time_offset

    data_ohlc = df_trades["price"].resample(resampling_frequency, offset=offset).ohlc()
    # we throw artificially inserted events
    df_trades = df_trades.iloc[1:-1, :]
    # if there was no trade in a given time frame, we use the following tactic:
    # open, low, high, close, weighted is set to the close value from the previous interval,
    # you can specify another column to fill in the gaps, e.g. weighted
    data_ohlc["weighted"] = df_trades.resample(
        resampling_frequency, offset=offset
    ).apply(lambda x: average(x.price, weights=x.qty))
    data_ohlc[fill_column].fillna(method="ffill", inplace=True, axis=0)
    data_ohlc.fillna(method="bfill", inplace=True, axis=1)
    data_ohlc.fillna(method="ffill", inplace=True, axis=1)

    data_ohlc["volume"] = (
        df_trades["qty"].resample(resampling_frequency, offset=offset).sum()
    )
    data_ohlc["volume_asset"] = (
        (df_trades["price"] * df_trades["qty"])
        .resample(resampling_frequency, offset=offset)
        .sum()
    )
    data_ohlc["volume_asset_buyer_maker"] = (
        (df_trades["price"] * df_trades["qty"] * df_trades["isBuyerMaker"])
        .resample(resampling_frequency, offset=offset)
        .sum()
    )
    data_ohlc["volume_asset_buyer_taker"] = (
        data_ohlc["volume_asset"] - data_ohlc["volume_asset_buyer_maker"]
    )
    data_ohlc["trades"] = (
        df_trades["tradeId"].resample(resampling_frequency, offset=offset).nunique()
    )
    data_ohlc["trades_full"] = (
        df_trades["timestamp"].resample(resampling_frequency, offset=offset).nunique()
    )
    data_ohlc.fillna(value=0, inplace=True)

    return data_ohlc


# Concatenate OHLC data from multiple days
def merge_ohlc(ohlc_data, volume_filter=True):
    def volume_filtered_agg(x, func):
        x_len = len(x)
        if x_len == 1:
            return x
        else:
            timestamp = x.index[0]
            volume = concat.loc[timestamp]["volume"].values
            x = x.values
            volume_filtered = [x[i] for i in range(x_len) if volume[i]]
            return func(volume_filtered)

    if isinstance(ohlc_data, list):
        concat = pd.concat(ohlc_data)
    elif isinstance(ohlc_data, dict):
        concat = pd.concat(
            [v for k, v in sorted(ohlc_data.items(), key=lambda item: item[0])]
        )
    else:
        raise ValueError(f"ohlc_data can be list or dict, but is {type(ohlc_data)}")

    # During aggregation, when we combine intervals from the border of a day from two dataframes, an error may occur.
    # Normally, when we want to find the low value, we take the value from the first interval and the close value from the last interval.
    # However, some intervals have been artificially created by propagating values from previous intervals
    # To have more precise results, we must first filter out the intervals, throwing out those where there is zero volume,
    # and only then use our aggregation functions, e.g. max or first. this method is more accurate but takes a little longer
    if volume_filter:
        ohlc_agg_dict = {
            "open": partial(volume_filtered_agg, func=lambda x: x[0]),
            "high": partial(volume_filtered_agg, func=lambda x: max(x)),
            "low": partial(volume_filtered_agg, func=lambda x: min(x)),
            "close": partial(volume_filtered_agg, func=lambda x: x[-1]),
            "volume": "sum",
            "volume_asset": "sum",
            "volume_asset_buyer_maker": "sum",
            "volume_asset_buyer_taker": "sum",
            "trades": "sum",
            "trades_full": "sum",
        }
    else:
        ohlc_agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "volume_asset": "sum",
            "volume_asset_buyer_maker": "sum",
            "volume_asset_buyer_taker": "sum",
            "trades": "sum",
            "trades_full": "sum",
        }

    weighted = concat.groupby(by="timestamp").apply(
        lambda x: average(x["weighted"], weights=x["volume"])
    )
    merged = concat.groupby(by="timestamp").agg(ohlc_agg_dict)
    merged.insert(4, "weighted", weighted)

    merged["volume_asset_buyer_taker_ratio"] = (
        merged["volume_asset_buyer_taker"] / merged["volume_asset"]
    )
    merged["volume_asset_buyer_taker_ratio"].fillna(method="ffill", inplace=True)

    return merged


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


# wraper on pandas concatenation catching error of concatenation of empty list and list containing none values
def safe_concat(dataframes):
    dataframes = [df for df in dataframes if df is not None]
    try:
        return pd.concat(dataframes)
    except ValueError:
        return None


def shift(xs, n):
    e = np.empty_like(xs)
    if n > 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    elif n == 0:
        e = xs
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def pct_change(array, columns_order, seq_len, base_column, future_column, future_steps):
    base = shift(array[:, np.where(columns_order == base_column)[0][0]], -seq_len - 1)
    future = array[:, np.where(columns_order == future_column)[0][0]]
    future = shift(future, -future_steps - seq_len - 1)
    pct_change = future / base - 1

    return pct_change, future_steps


def raw_future_value(array, columns_order, seq_len, future_column, future_steps):
    future = array[:, np.where(columns_order == future_column)[0][0]]
    future = shift(future, -future_steps - seq_len + 1)

    return future, future_steps


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


# custom encoder for writing numpy arrays to a json file
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        else:
            return super(NpEncoder, self).default(obj)


class OHLCDataSet:

    """Class that enables the generation of datasets,
    with we can wrap the entire process of building a dataset of Binance trading data with technical analysis.

    Example:
    ```ods = OHLCDataSet(
    pairs=[('eth', 'usdt'), ('btc', 'usdt')],
    start_date=date(2022, 3, 10),
    end_date=date(2022, 3, 13),
    train_end=date(2022, 3, 12),
    ohlc_intervals = [2, 4, 10],
    multithreading=True,
    )
    ```
    Args:
        pairs: List of trading pairs symbol.
        start_date: Date From when to collect data.
        end_date: Date by which data is collected.
        train_end. The date when we consider that the training set ends, important because some global statistics,
        such as the average, are calculated only on the basis of training data.
        base_time_offset: Time unit which constitutes the database of the data set.
        All the time intervals used must be an integer multiple of this value.
        ohlc_intervals: Defines with which time step size the time series are created,
        multiples of the base unit base_time_offset.
        drop_n_first_rows: How many leading rows are thrown from the DataFrame,
        the values may be different for different intervals
        dataset_seq_len: Defines the number of time steps that the final samples used to train the models will have.
        The length of the time series can be different for different intervals.
        drop_columns_names: The names of the columns that we do not want to use may be different for different intervals
        columns: a dictionary containing the following keys:
        monotonic_col, price_col, pair_col, zero_one_col, candle_col, other_col
        which contain assignments of particular column names to different classes of features.
        If one is not provided, or the entire dictionary is not provided,
        the assignment will be generated automatically.
        If you do not know your data well, it is better to leave it blank.
        multithreading: Use of multithreading for data retrieval and preprocessing
        transform_mode: Specifies the procedure of normalization of features not correlated with the price (often fat-tail distributions)
        The choices are ‘power’, ‘quintile’ or ‘mix’ (default).
        ‘power’ uses PowerTransformer, ‘quintile’ uses QuantileTransformer and ‘mix’ uses PowerTransformer
        unless there is an error then uses QuantileTransformer.
        binance_base_url: url for binance data, by default 'https://data.binance.vision'.
        data_dir: Where to save downloaded files, by default 'data'
        fill_gaps_limit: removes columns that have gaps longer than fill_gaps_limit, by default 5
        inverse_error_quintiles: the quintile values ​​for which the inverse transformation error is to be calculated,
        by default [q for q in np.arange(0,1,0.1)] + [0.95, 0.99, 0.999]


    """

    def __init__(
        self,
        pairs,
        start_date,
        end_date,
        train_end,
        base_time_offset=pd.tseries.offsets.Minute(),
        price_scaling_method="standardised_max_pct_change",
        ohlc_intervals=None,
        drop_n_first_rows=None,
        dataset_seq_len=None,
        drop_columns_names=None,
        columns=None,
        multithreading=False,
        keep_unchanged_df=False,
        transform_mode="mix",
        binance_base_url="https://data.binance.vision",
        data_dir="data",
        fill_gaps_limit=5,
        inverse_error_quintiles=None,
    ):
        self.inverse_error_quintiles = inverse_error_quintiles
        if self.inverse_error_quintiles is None:
            self.inverse_error_quintiles = [q for q in np.arange(0, 1, 0.1)] + [
                0.95,
                0.99,
                0.999,
            ]
        self.binance_base_url = binance_base_url
        self.data_dir = data_dir
        self.fill_gaps_limit = fill_gaps_limit
        if ohlc_intervals is None:
            ohlc_intervals = [1, 10, 60]
        for interval in ohlc_intervals[1:]:
            if interval % ohlc_intervals[0]:
                raise ValueError(
                    "successive intervals must be integers of multiples of the basic interval"
                )
        if drop_n_first_rows is None:
            drop_n_first_rows = {interval: 96 for interval in ohlc_intervals}
        if dataset_seq_len is None:
            dataset_seq_len = {interval: 30 for interval in ohlc_intervals}
        if drop_columns_names is None:
            drop_columns_names = {
                interval: [
                    "DPO_20",
                    "ICS_26",
                    "EOM_14_100000000",
                    "QQEl_14_5_4.236",
                    "QQEs_14_5_4.236",
                    "SQZPRO_NO",
                    "SQZ_NO",
                ]
                for interval in ohlc_intervals
            }
        if columns is None:
            columns = {}

        self.pairs = [asset + base for asset, base in pairs]
        self.asset_base_pairs = pairs

        self.start_date = start_date
        self.end_date = end_date
        self.train_end = train_end
        self.base_time_offset = base_time_offset
        self.price_scaling_method = price_scaling_method
        self.ohlc_intervals = ohlc_intervals
        self.drop_n_first_rows = drop_n_first_rows
        self.dataset_seq_len = dataset_seq_len
        self.drop_columns_names = drop_columns_names

        # You can provide dict that describes different type of columns, or this can be calculated from data
        self.monotonic_col = columns.get("monotonic_col")
        self.price_col = columns.get("price_col")
        # this one set of columns is an exception, it must be specified before preprocessing
        # because we use it at an early stage of data cleansing which is needed to calculated the rest of the column types
        self.pair_col = columns.get(
            "pair_col",
            [
                "HILOl_13_21",
                "HILOs_13_21",
                "PSARl_0.02_0.2",
                "PSARs_0.02_0.2",
                "QQEl_14_5_4.236",
                "QQEs_14_5_4.236",
                "SUPERTl_7_3.0",
                "SUPERTs_7_3.0",
            ],
        )
        self.time_col = [
            "HOUR_SIN",
            "HOUR_COS",
            "DAY_SIN",
            "DAY_COS",
            "WEEK_SIN",
            "WEEK_COS",
            "YEAR_SIN",
            "YEAR_COS",
        ]
        self.zero_one_col = columns.get("zero_one_col")
        self.candle_col = columns.get("candle_col")
        self.other_col = columns.get("other_col")

        # dataset parameters, set in self.make_dataset
        self.all_columns = None
        self.power_factors = None
        self.price_std = None

        self.ohlc_data = {
            pair: {
                interval: {
                    self.ohlc_intervals[0] * step: {}
                    for step in range(interval // self.ohlc_intervals[0])
                }
                for interval in self.ohlc_intervals
            }
            for pair in self.pairs
        }

        self.multithreading = multithreading
        self.keep_unchanged_df = keep_unchanged_df
        self.unchanged_df = None
        self.transform_mode = transform_mode
        # Dictionary to store the time taken by individual tasks
        self.timing = {}

    def print_lenght(self):
        i = 0
        for interval in self.ohlc_intervals:
            for step in range(interval // self.ohlc_intervals[0]):
                for pair in self.pairs:
                    print(
                        i,
                        pair,
                        interval,
                        step,
                        self.ohlc_data[pair][interval][self.ohlc_intervals[0] * step]
                        .to_numpy(dtype=np.float64)
                        .shape,
                        self.ohlc_data[pair][interval][
                            self.ohlc_intervals[0] * step
                        ].index.nunique(),
                    )
                    i += 1

    def make_dataset(self):
        # loading and basic cleaning data
        start = monotonic()
        start_make_dataset = start
        self._load_data()
        self.timing["load_time"] = monotonic() - start

        start = monotonic()
        self._clear_data()
        self.timing["clear_time"] = monotonic() - start
        start = monotonic()
        self._remove_no_variance_features()
        self.timing["remove_no_variance_time"] = monotonic() - start
        start = monotonic()
        self._remove_redundant_features()
        self.timing["remove_redundant_features_time"] = monotonic() - start

        start = monotonic()
        # organizing features by various types (various preprocessing for various types)
        self.all_columns = self.get_all_columns()
        if self.monotonic_col is None:
            self.monotonic_col = self.get_monotonic_col()
        if self.zero_one_col is None:
            self.zero_one_col = self.get_zero_one_col()
        if self.candle_col is None:
            self.candle_col = self.get_candle_col()
        self.pair_col = self.get_pairs_col()
        start_inner = monotonic()
        if self.price_col is None:
            self.price_col = self.get_price_like_col()
        self.timing["get_price_columns_time"] = monotonic() - start_inner
        self.timing["get_columns_time"] = monotonic() - start

        start = monotonic()
        self.monotonic_col = [
            col for col in self.monotonic_col if col not in self.candle_col
        ]
        self.pair_col = [col for col in self.pair_col if col not in self.candle_col]
        self.zero_one_col = [
            col for col in self.zero_one_col if col not in self.candle_col
        ]
        self.price_col = [
            col
            for col in self.price_col
            if all(
                [
                    col not in cols
                    for cols in [
                        self.monotonic_col,
                        self.pair_col,
                        self.candle_col,
                        self.zero_one_col,
                    ]
                ]
            )
        ]
        self.other_col = [
            col
            for col in self.all_columns
            if all(
                [
                    col not in cols
                    for cols in [
                        self.monotonic_col,
                        self.pair_col,
                        self.candle_col,
                        self.zero_one_col,
                        self.price_col,
                    ]
                ]
            )
        ]
        self.timing["columns_organizing_time"] = monotonic() - start
        self._drop_first_unused()
        self._trim_longer_dataframes(front_trim=False)

        self.timing["drop_unused_time"] = monotonic() - start
        if self.keep_unchanged_df:
            self.unchanged_df = copy.deepcopy(self.ohlc_data)

        start = monotonic()
        self._drop_monotonic_col()
        self.timing["drop_monotonic_time"] = monotonic() - start

        start = monotonic()
        self._normalize_candle_col()
        self.timing["normalize_candle_time"] = monotonic() - start

        start = monotonic()
        self.power_factors = self._normalize_other_col(mode=self.transform_mode)
        self.timing["normalize_other_time"] = monotonic() - start

        start = monotonic()
        self.price_std = self.get_price_std()
        self.timing["get_price_std_time"] = monotonic() - start

        start = monotonic()
        self._add_time_features()
        self.all_columns.extend(self.time_col)
        self.timing["add_time_features_time"] = monotonic() - start
        self.timing["whole_process_time"] = monotonic() - start_make_dataset

    def _process_single_data(self, pair, single_date, verbose=False):
        if verbose:
            print(pair, single_date.strftime("%Y-%m-%d"))
        try:
            day_trades = get_trades(
                pair,
                single_date.year,
                str(single_date.month).zfill(2),
                str(single_date.day).zfill(2),
                binance_base_url=self.binance_base_url,
                trades_dir=join(self.data_dir, "trades"),
            )
        except zipfile.BadZipfile:
            print("BadZipfile, probably wrong date")
            return None

        for interval in self.ohlc_intervals:
            for step in range(interval // self.ohlc_intervals[0]):
                offset = self.ohlc_intervals[0] * step
                day_ohlc = trades2ohlc(
                    day_trades,
                    resampling_frequency=interval,
                    offset=offset,
                    base_time_offset=self.base_time_offset,
                )
                self.ohlc_data[pair][interval][offset][single_date] = day_ohlc

    def _process_ohlc_ta(self, pair, interval, offset, save_to_csv=False):
        self.ohlc_data[pair][interval][offset] = merge_ohlc(
            self.ohlc_data[pair][interval][offset]
        )
        self.ohlc_data[pair][interval][offset].ta.strategy(ta.AllStrategy)

        # remove don`t used collumns
        for col in self.drop_columns_names[interval]:
            self.ohlc_data[pair][interval][offset].pop(col)

        # Some technical indicators needs some previous data points to calculate value,
        # by dropping first n rows we put more features in our dataset.
        self.ohlc_data[pair][interval][offset] = self.ohlc_data[pair][interval][
            offset
        ].iloc[self.drop_n_first_rows[interval] :]
        # remove columns that are still empty at the beginning after the set number of leading
        # rows have been dropped. Also removes columns that have gaps longer than self.fill_gaps_limit (5)
        # These infinities are a numerical error, when calculating technical indicators,
        # we want to treat them as not a number
        self.ohlc_data[pair][interval][offset].replace(
            [np.inf, -np.inf], np.nan, inplace=True
        )
        for col in self.ohlc_data[pair][interval][offset]:
            if col in self.pair_col:
                continue
            self.ohlc_data[pair][interval][offset].fillna(
                method="ffill", limit=self.fill_gaps_limit, inplace=True
            )
            if self.ohlc_data[pair][interval][offset][col].isnull().values.any():
                self.ohlc_data[pair][interval][offset].pop(col)

    def _load_data(self):
        pd_list = [
            (pair, single_date)
            for pair in self.pairs
            for single_date in daterange(self.start_date, self.end_date)
        ]
        params_list = [
            (pair, interval, self.ohlc_intervals[0] * step)
            for pair in self.pairs
            for interval in self.ohlc_intervals
            for step in range(interval // self.ohlc_intervals[0])
        ]

        if self.multithreading:
            with ThreadPoolExecutor(20) as executor:
                futures = [
                    executor.submit(self._process_single_data, *pair_data)
                    for pair_data in pd_list
                ]
                wait(futures, return_when=ALL_COMPLETED)
                futures = [
                    executor.submit(self._process_ohlc_ta, *params)
                    for params in params_list
                ]
                wait(futures, return_when=ALL_COMPLETED)
        else:
            for pair_data in pd_list:
                self._process_single_data(*pair_data)
            for params in params_list:
                self._process_ohlc_ta(*params)

    def _drop_monotonic_col(self):
        for pair in self.pairs:
            for interval in self.ohlc_intervals:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    for col in self.ohlc_data[pair][interval][offset]:
                        if col in self.monotonic_col:
                            self.ohlc_data[pair][interval][offset].pop(col)

    # For the same pair and the same dataframe interval, they should be the same length for all offsets.
    # Mainly due to later vectorization and performance issues
    def _trim_longer_dataframes(self, front_trim=True):
        for pair in self.pairs:
            for interval in self.ohlc_intervals[1:]:
                min_sequence_len = min(
                    [
                        len(
                            self.ohlc_data[pair][interval][
                                self.ohlc_intervals[0] * step
                            ]
                        )
                        for step in range(interval // self.ohlc_intervals[0])
                    ]
                )
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    if len(self.ohlc_data[pair][interval][offset]) > min_sequence_len:
                        to_drop = (
                            len(self.ohlc_data[pair][interval][offset])
                            - min_sequence_len
                        )
                        if to_drop > 0:
                            if front_trim:
                                to_drop = self.ohlc_data[pair][interval][offset].index[
                                    :to_drop
                                ]
                                self.ohlc_data[pair][interval][offset].drop(
                                    index=to_drop, inplace=True
                                )
                            else:
                                to_drop = self.ohlc_data[pair][interval][offset].index[
                                    -to_drop:
                                ]
                                self.ohlc_data[pair][interval][offset].drop(
                                    index=to_drop, inplace=True
                                )

    def _clear_data(self):
        # If some pair (we allow different sets of features for different intervals) dont have some column,
        # this column also must be dropped from others dataframes
        for interval in self.ohlc_intervals:
            columns_lists = []
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    columns_lists.append(
                        self.ohlc_data[pair][interval][offset].columns.values
                    )
            all_columns = []
            [all_columns.extend(cols) for cols in columns_lists]
            all_columns = list(set(all_columns))
            common_columns = [
                c for c in all_columns if all([c in cc for cc in columns_lists])
            ]

            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    for col in self.ohlc_data[pair][interval][offset]:
                        if col not in common_columns:
                            self.ohlc_data[pair][interval][offset].pop(col)

        self._trim_longer_dataframes()
        # Some time steps will never be used, so they can be deleted
        for pair in self.pairs:
            last_base_timestamp = self.ohlc_data[pair][self.ohlc_intervals[0]][0].index[
                -1
            ]
            first_base_timestamp = self.ohlc_data[pair][self.ohlc_intervals[0]][
                0
            ].index[0]
            first_usable_timestamp = first_base_timestamp

            for interval in self.ohlc_intervals[1:]:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    self.ohlc_data[pair][interval][offset] = self.ohlc_data[pair][
                        interval
                    ][offset][
                        (
                            self.ohlc_data[pair][interval][offset].index
                            + (interval - self.ohlc_intervals[0])
                            * self.base_time_offset
                            <= last_base_timestamp
                        )
                        & (
                            self.ohlc_data[pair][interval][offset].index
                            >= first_base_timestamp
                        )
                    ]
                first_usable_timestamp = max(
                    self.ohlc_data[pair][interval][self.ohlc_intervals[0]].index[0]
                    + self.dataset_seq_len[interval] * interval * self.base_time_offset,
                    first_usable_timestamp,
                )

            self.ohlc_data[pair][self.ohlc_intervals[0]][0] = self.ohlc_data[pair][
                self.ohlc_intervals[0]
            ][0][
                self.ohlc_data[pair][self.ohlc_intervals[0]][0].index
                >= first_usable_timestamp
            ]
            first_base_timestamp = self.ohlc_data[pair][self.ohlc_intervals[0]][
                0
            ].index[0]
            for interval in self.ohlc_intervals[1:]:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    self.ohlc_data[pair][interval][offset] = self.ohlc_data[pair][
                        interval
                    ][offset][
                        self.ohlc_data[pair][interval][offset].index
                        + self.dataset_seq_len[interval]
                        * interval
                        * self.base_time_offset
                        >= first_base_timestamp
                    ]

    def _remove_no_variance_features(self):
        for interval in self.ohlc_intervals:

            columns_lists = []
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    columns_lists.append(
                        self.ohlc_data[pair][interval][offset].columns.values
                    )
            all_columns = []
            [all_columns.extend(cols) for cols in columns_lists]
            all_columns = list(set(all_columns))

            no_variance_columns = []
            for col in all_columns:
                col_values = []
                for pair in self.pairs:
                    for step in range(interval // self.ohlc_intervals[0]):
                        offset = self.ohlc_intervals[0] * step
                        col_values.extend(
                            self.ohlc_data[pair][interval][offset][col].unique()
                        )
                col_values = list(set(col_values))
                if len(col_values) <= 1:
                    no_variance_columns.append(col)

            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    for col in no_variance_columns:
                        self.ohlc_data[pair][interval][offset].pop(col)

        return self.ohlc_data

    def _remove_redundant_features(self):
        for interval in self.ohlc_intervals:
            cor = (
                pd.concat(
                    [
                        pd.concat(
                            [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ]
                                for pair in self.pairs
                            ]
                        )
                        for step in range(interval // self.ohlc_intervals[0])
                    ]
                )
                .corr()
                .abs()
            )

            upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
            to_drop = [
                column for column in upper_tri.columns if any(upper_tri[column] == 1)
            ]

            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    self.ohlc_data[pair][interval][offset].drop(
                        to_drop, axis=1, inplace=True
                    )

    def get_all_columns(self):
        all_columns = []
        [
            all_columns.extend(
                self.ohlc_data[self.pairs[0]][interval][0].columns.values
            )
            for interval in self.ohlc_intervals
        ]
        return sorted(list(set(all_columns)))

    def get_monotonic_col(self):

        monotonic_features = {}
        for interval in self.ohlc_intervals:
            for feature in self.all_columns:
                if feature in self.ohlc_data[self.pairs[0]][interval][0]:
                    is_monotonic = True
                    for pair in self.pairs:
                        for step in range(interval // self.ohlc_intervals[0]):
                            offset = self.ohlc_intervals[0] * step
                            is_monotonic = (
                                self.ohlc_data[pair][interval][offset][
                                    feature
                                ].is_monotonic
                                or self.ohlc_data[pair][interval][offset][
                                    feature
                                ].is_monotonic_decreasing
                            )
                            if not is_monotonic:
                                break
                        if not is_monotonic:
                            break
                    if is_monotonic:
                        monotonic_features.setdefault(feature, []).append(True)
                    else:
                        monotonic_features.setdefault(feature, []).append(False)

        monotonic_features = [
            feature
            for feature, is_monotonic in monotonic_features.items()
            if all(is_monotonic)
        ]
        return monotonic_features

    def get_price_like_col(self, price_col="close", price_treshold=0.55):
        price_series = pd.concat(
            [
                pd.concat(
                    [
                        pd.concat(
                            [
                                self.ohlc_data[pair][interval][offset][price_col]
                                for offset in [
                                    self.ohlc_intervals[0] * step
                                    for step in range(
                                        interval // self.ohlc_intervals[0]
                                    )
                                ]
                                if price_col in self.ohlc_data[pair][interval][offset]
                            ]
                        )
                        for pair in self.pairs
                    ]
                )
                for interval in self.ohlc_intervals
            ]
        )

        col_stats = {}
        for col in self.all_columns:
            if col.startswith("CDL_"):
                continue
            col_series = pd.concat(
                [
                    pd.concat(
                        [
                            pd.concat(
                                [
                                    self.ohlc_data[pair][interval][offset][col]
                                    if col in self.ohlc_data[pair][interval][offset]
                                    else pd.Series(
                                        [
                                            None
                                            for _ in range(
                                                len(
                                                    self.ohlc_data[pair][interval][
                                                        offset
                                                    ]
                                                )
                                            )
                                        ],
                                        name=col,
                                        dtype=float,
                                        index=self.ohlc_data[pair][interval][
                                            offset
                                        ].index,
                                    )
                                    for offset in [
                                        self.ohlc_intervals[0] * step
                                        for step in range(
                                            interval // self.ohlc_intervals[0]
                                        )
                                    ]
                                ]
                            )
                            for pair in self.pairs
                        ]
                    )
                    for interval in self.ohlc_intervals
                ]
            )
            col_stats[col] = {
                "mean": (col_series / price_series).mean(),
                "std": (col_series / price_series).std(),
            }

        price_features = [
            col
            for col in col_stats.keys()
            if abs(col_stats[col]["mean"] - 1) <= price_treshold
            and col_stats[col]["std"] <= price_treshold
        ]
        return price_features

    def get_candle_col(self):
        candle_col = []
        for col in self.all_columns:
            if col.startswith("CDL_"):
                candle_col.append(col)
        return candle_col

    def get_zero_one_col(self):
        zero_one_features = []
        for col in self.all_columns:
            if col.startswith("CDL_"):
                continue
            col_series = safe_concat(
                [
                    safe_concat(
                        [
                            safe_concat(
                                [
                                    self.ohlc_data[pair][interval][offset][col]
                                    for offset in [
                                        self.ohlc_intervals[0] * step
                                        for step in range(
                                            interval // self.ohlc_intervals[0]
                                        )
                                    ]
                                    if col in self.ohlc_data[pair][interval][offset]
                                ]
                            )
                            for pair in self.pairs
                        ]
                    )
                    for interval in self.ohlc_intervals
                ]
            )
            if col_series.isin([-1, 0, 1]).all():
                zero_one_features.append(col)
        return zero_one_features

    def get_pairs_col(self):
        pairs = [
            "HILOl_13_21",
            "HILOs_13_21",
            "PSARl_0.02_0.2",
            "PSARs_0.02_0.2",
            "QQEl_14_5_4.236",
            "QQEs_14_5_4.236",
            "SUPERTl_7_3.0",
            "SUPERTs_7_3.0",
        ]
        return [col for col in self.all_columns if col in pairs]

    def _normalize_candle_col(self):
        for interval in self.ohlc_intervals:
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    candle_col_exist = [
                        col
                        for col in self.candle_col
                        if col in self.ohlc_data[pair][interval][offset]
                    ]
                    self.ohlc_data[pair][interval][offset][candle_col_exist] /= 100

    def _normalize_other_col(self, mode="mix"):
        if mode == "power":
            return self._power_transform_other_col()
        elif mode == "quintile":
            return self._quintile_other_col()
        elif mode == "mix":
            return self._power_quintile_transform_other_col()
        else:
            raise ValueError(f"_normalize_other_col, mode {mode} not supported")

    def _quintile_other_col(self):
        result = {}
        for interval in self.ohlc_intervals:
            other_columns = [
                c
                for c in self.ohlc_data[self.pairs[0]][interval][0].columns.values
                if c in self.other_col
            ]
            sc_quint = QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                ignore_implicit_zeros=False,
                subsample=100000,
            )
            sc_quint.fit(
                pd.concat(
                    [
                        pd.concat(
                            [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ].loc[self.start_date : self.train_end][other_columns]
                                for pair in self.pairs
                            ]
                        )
                        for step in range(interval // self.ohlc_intervals[0])
                    ]
                )
            )
            errors = []
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    transformed = sc_quint.transform(
                        self.ohlc_data[pair][interval][offset][other_columns]
                    )
                    restored = sc_quint.inverse_transform(transformed)
                    original = self.ohlc_data[pair][interval][offset][
                        other_columns
                    ].to_numpy()
                    inverse_error = restored - original
                    inverse_error = np.absolute(
                        np.divide(
                            inverse_error,
                            original,
                            out=np.zeros_like(inverse_error),
                            where=original != 0,
                        )
                    )
                    errors.append(inverse_error)
                    self.ohlc_data[pair][interval][offset][other_columns] = transformed

            errors = np.concatenate(errors, axis=0)
            max_errors = errors.max(axis=0)
            errors = np.array(
                [
                    [
                        np.quantile(errors, q_val, axis=0)
                        for q_val in self.inverse_error_quintiles
                    ]
                ]
            ).T
            errors = errors.reshape((errors.shape[0], -1))

            result[interval] = {
                name: {
                    "quantiles": quantiles_,
                    "mode": "quantiles",
                    "inverse_error_quantiles": q_error,
                    "max_inverse_error": m_error,
                }
                for name, quantiles_, q_error, m_error in zip(
                    other_columns, sc_quint.quantiles_.T, errors, max_errors
                )
            }
            result[interval]["used_quintiles"] = self.inverse_error_quintiles
        return result

    def _power_quintile_transform_other_col(self):
        result = {}
        for interval in self.ohlc_intervals:
            result[interval] = {}

            other_columns = [
                c
                for c in self.ohlc_data[self.pairs[0]][interval][0].columns.values
                if c in self.other_col
            ]

            sc_quint = QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                ignore_implicit_zeros=False,
                subsample=100000,
            )
            sc_power = PowerTransformer(method="yeo-johnson")

            for i, column_name in enumerate(other_columns):
                merged_data = pd.concat(
                    [
                        pd.concat(
                            [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ].loc[self.start_date : self.train_end][[column_name]]
                                for pair in self.pairs
                            ]
                        )
                        for step in range(interval // self.ohlc_intervals[0])
                    ]
                )

                sc_quint.fit(merged_data)
                sc_power.fit(merged_data)

                errors = []
                if sc_power._scaler.var_[0] > 0:
                    for pair in self.pairs:
                        for step in range(interval // self.ohlc_intervals[0]):
                            offset = self.ohlc_intervals[0] * step
                            transformed = sc_power.transform(
                                self.ohlc_data[pair][interval][offset][[column_name]]
                            )
                            restored = sc_power.inverse_transform(transformed)
                            original = self.ohlc_data[pair][interval][offset][
                                [column_name]
                            ].to_numpy()
                            inverse_error = restored - original
                            inverse_error = np.absolute(
                                np.divide(
                                    inverse_error,
                                    original,
                                    out=np.zeros_like(inverse_error),
                                    where=original != 0,
                                )
                            )
                            errors.append(inverse_error)
                            self.ohlc_data[pair][interval][offset][
                                [column_name]
                            ] = transformed

                    eshapes = [e.shape for e in errors]
                    errors = np.concatenate(errors, axis=0)
                    result[interval][column_name] = {
                        "mean": sc_power._scaler.mean_,
                        "var": sc_power._scaler.var_,
                        "lambda": sc_power.lambdas_,
                        "mode": "power",
                        "inverse_error_quantiles": {
                            q_val: np.quantile(errors, q_val)
                            for q_val in self.inverse_error_quintiles
                        },
                        "max_inverse_error": errors.max(),
                    }

                else:
                    for pair in self.pairs:
                        for step in range(interval // self.ohlc_intervals[0]):
                            offset = self.ohlc_intervals[0] * step
                            transformed = sc_quint.transform(
                                self.ohlc_data[pair][interval][offset][[column_name]]
                            )
                            restored = sc_quint.inverse_transform(transformed)
                            original = self.ohlc_data[pair][interval][offset][
                                [column_name]
                            ].to_numpy()
                            inverse_error = np.absolute(restored - original)
                            inverse_error = np.divide(
                                inverse_error,
                                original,
                                out=np.zeros_like(inverse_error),
                                where=original != 0,
                            )
                            errors.append(inverse_error)
                            self.ohlc_data[pair][interval][offset][
                                [column_name]
                            ] = transformed

                    eshapes = [e.shape for e in errors]
                    errors = np.concatenate(errors, axis=0)
                    result[interval][column_name] = {
                        "quantiles": sc_quint.quantiles_,
                        "mode": "quantiles",
                        "inverse_error_quantiles": {
                            q_val: np.quantile(errors, q_val)
                            for q_val in self.inverse_error_quintiles
                        },
                        "max_inverse_error": errors.max(),
                    }

        return result

    def _power_transform_other_col(self):
        result = {}
        for interval in self.ohlc_intervals:
            other_columns = [
                c
                for c in self.ohlc_data[self.pairs[0]][interval][0].columns.values
                if c in self.other_col
            ]
            sc_power = PowerTransformer()
            sc_power.fit(
                pd.concat(
                    [
                        pd.concat(
                            [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ].loc[self.start_date : self.train_end][other_columns]
                                for pair in self.pairs
                            ]
                        )
                        for step in range(interval // self.ohlc_intervals[0])
                    ]
                )
            )

            errors = []
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step
                    transformed = sc_power.transform(
                        self.ohlc_data[pair][interval][offset][other_columns]
                    )
                    restored = sc_power.inverse_transform(transformed)
                    original = self.ohlc_data[pair][interval][offset][
                        other_columns
                    ].to_numpy()
                    inverse_error = restored - original
                    inverse_error = np.absolute(
                        np.divide(
                            inverse_error,
                            original,
                            out=np.zeros_like(inverse_error),
                            where=original != 0,
                        )
                    )
                    errors.append(inverse_error)
                    self.ohlc_data[pair][interval][offset][other_columns] = transformed

            errors = np.concatenate(errors, axis=0)
            max_errors = errors.max(axis=0)
            errors = np.array(
                [
                    [
                        np.quantile(errors, q_val, axis=0)
                        for q_val in self.inverse_error_quintiles
                    ]
                ]
            ).T
            errors = errors.reshape((errors.shape[0], -1))

            result[interval] = {
                name: {
                    "mean": mean_,
                    "var": var_,
                    "lambda": lambda_,
                    "mode": "power",
                    "inverse_error_quantiles": q_error,
                    "max_inverse_error": m_error,
                }
                for name, mean_, var_, lambda_, q_error, m_error in zip(
                    other_columns,
                    sc_power._scaler.mean_,
                    sc_power._scaler.var_,
                    sc_power.lambdas_,
                    errors,
                    max_errors,
                )
            }
            result[interval]["used_quintiles"] = self.inverse_error_quintiles

        return result

    def get_price_std(self):

        pct_change_result = {}
        max_pct_change_result = {}
        last_pct_change_result = {}
        price_cols = ["open", "low", "high", "close", "weighted"]
        for price_col in price_cols:
            pct_change_result[price_col] = {
                interval: pd.concat(
                    [
                        pd.DataFrame().assign(
                            **{
                                str(i): self.ohlc_data[pair][interval][offset]
                                .loc[self.start_date : self.train_end][price_col]
                                .pct_change(periods=-i)
                                for i in range(1, self.dataset_seq_len[interval])
                            }
                        )
                        for pair in self.pairs
                        for offset in [
                            self.ohlc_intervals[0] * step
                            for step in range(interval // self.ohlc_intervals[0])
                        ]
                    ]
                )
                .stack()
                .std()
                for interval in self.ohlc_intervals
            }

            max_pct_change_result[price_col] = {
                interval: pd.concat(
                    [
                        pd.DataFrame().assign(
                            **{
                                str(i): self.ohlc_data[pair][interval][offset]
                                .loc[self.start_date : self.train_end][price_col]
                                .pct_change(periods=-i)
                                for i in range(1, self.dataset_seq_len[interval])
                            }
                        )
                        for pair in self.pairs
                        for offset in [
                            self.ohlc_intervals[0] * step
                            for step in range(interval // self.ohlc_intervals[0])
                        ]
                    ]
                )
                .abs()
                .max(axis=1)
                .std()
                for interval in self.ohlc_intervals
            }

            last_pct_change_result[price_col] = {
                interval: pd.concat(
                    [
                        self.ohlc_data[pair][interval][offset]
                        .loc[self.start_date : self.train_end][price_col]
                        .pct_change(periods=self.dataset_seq_len[interval] - 1)
                        for pair in self.pairs
                        for offset in [
                            self.ohlc_intervals[0] * step
                            for step in range(interval // self.ohlc_intervals[0])
                        ]
                    ]
                ).std()
                for interval in self.ohlc_intervals
            }

        result = {
            interval: {
                price_col: {
                    "pct_change": pct_change_result[price_col][interval],
                    "max_pct_change": max_pct_change_result[price_col][interval],
                    "last_pct_change": last_pct_change_result[price_col][interval],
                }
                for price_col in price_cols
            }
            for interval in self.ohlc_intervals
        }

        return result

    def _drop_first_unused(self):
        for interval in self.ohlc_intervals[1:]:
            freq = interval // self.ohlc_intervals[0]
            offsets = [
                self.ohlc_intervals[0] * step
                for step in range(interval // self.ohlc_intervals[0])
            ]
            offsets = sorted(offsets, reverse=True)[:-1]
            offsets.insert(0, 0)
            for pair in self.pairs:
                base_list_len = len(self.ohlc_data[pair][self.ohlc_intervals[0]][0])
                first_n_pos = list(
                    range(
                        base_list_len
                        - self.dataset_seq_len[self.ohlc_intervals[0]]
                        + 1,
                        base_list_len
                        - self.dataset_seq_len[self.ohlc_intervals[0]]
                        + 1
                        - freq,
                        -1,
                    )
                )
                for pos in first_n_pos:
                    offset = offsets[(pos - 1) % freq]
                    idx = -((pos + freq - 1) // freq)
                    to_drop = (
                        len(self.ohlc_data[pair][interval][offset])
                        + idx
                        - self.dataset_seq_len[interval]
                        + 1
                    )
                    self.ohlc_data[pair][interval][offset].drop(
                        index=self.ohlc_data[pair][interval][offset].index[:to_drop],
                        inplace=True,
                    )

    # The time in seconds is not a useful feature. I have assumed a few fairly standard periodicity that may be present
    def _add_time_features(self, add_timestamp=False):
        hour = 60 * 60
        day = 24 * hour
        week = 7 * day
        year = (365.2425) * day

        for interval in self.ohlc_intervals:
            for pair in self.pairs:
                for step in range(interval // self.ohlc_intervals[0]):
                    offset = self.ohlc_intervals[0] * step

                    # now we dont assume intervals shorther than 1s sow we divide by 10**9 to convert nanoseconds to seconds
                    # it should be changed if we want to use shorter interwals.
                    # we add half of base_time_offset to get time of
                    timestamp_middle = (
                        self.ohlc_data[pair][interval][offset].index.asi8
                        + interval * self.base_time_offset.nanos / 2
                    )
                    seconds = timestamp_middle / 10**9

                    self.ohlc_data[pair][interval][offset]["HOUR_SIN"] = np.sin(
                        seconds * (2 * np.pi / hour)
                    )
                    self.ohlc_data[pair][interval][offset]["HOUR_COS"] = np.cos(
                        seconds * (2 * np.pi / hour)
                    )
                    self.ohlc_data[pair][interval][offset]["DAY_SIN"] = np.sin(
                        seconds * (2 * np.pi / day)
                    )
                    self.ohlc_data[pair][interval][offset]["DAY_COS"] = np.cos(
                        seconds * (2 * np.pi / day)
                    )
                    self.ohlc_data[pair][interval][offset]["WEEK_SIN"] = np.sin(
                        seconds * (2 * np.pi / week)
                    )
                    self.ohlc_data[pair][interval][offset]["WEEK_COS"] = np.cos(
                        seconds * (2 * np.pi / week)
                    )
                    self.ohlc_data[pair][interval][offset]["YEAR_SIN"] = np.sin(
                        seconds * (2 * np.pi / year)
                    )
                    self.ohlc_data[pair][interval][offset]["YEAR_COS"] = np.cos(
                        seconds * (2 * np.pi / year)
                    )

                    if add_timestamp:
                        self.ohlc_data[pair][interval][offset][
                            "TIMESTAMP"
                        ] = timestamp_middle

    def get_ohlc_data(self):
        return self.ohlc_data

    def get_arrays(self):

        base = np.concatenate(
            [
                self.ohlc_data[pair][self.ohlc_intervals[0]][0].to_numpy(
                    dtype=np.float64
                )
                for pair in self.pairs
            ]
        )

        other_intervals = [
            np.stack(
                [
                    np.concatenate(
                        [
                            self.ohlc_data[pair][interval][
                                self.ohlc_intervals[0] * step
                            ].to_numpy(dtype=np.float64)
                            for pair in self.pairs
                        ]
                    )
                    for step in range(interval // self.ohlc_intervals[0])
                ]
            )
            for interval in self.ohlc_intervals[1:]
        ]

        return base, *other_intervals

    def get_metadata(self):

        metadata = {
            "timing": self.timing,
            "pairs": self.pairs,
            "asset_base_pairs": self.asset_base_pairs,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "train_end": self.train_end,
            "base_time_offset": self.base_time_offset.__class__.__name__,
            "ohlc_intervals": self.ohlc_intervals,
            "drop_n_first_rows": self.drop_n_first_rows,
            "dataset_seq_len": self.dataset_seq_len,
            "drop_columns_names": self.drop_columns_names,
            "all_columns": self.all_columns,
            "monotonic_col": self.monotonic_col,
            "price_col": self.price_col,
            "pair_col": self.pair_col,
            "zero_one_col": self.zero_one_col,
            "candle_col": self.candle_col,
            "other_col": self.other_col,
            "dataframe_stats": {
                interval: {
                    "columns_order": self.ohlc_data[self.pairs[0]][interval][
                        0
                    ].columns.values,
                    "offsets": [
                        self.ohlc_intervals[0] * step
                        for step in range(interval // self.ohlc_intervals[0])
                    ],
                    "power_factors": self.power_factors[interval],
                    "price_std": self.price_std[interval],
                    "pairs": {
                        pair: {
                            "first_timestamps": [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ].index[0]
                                for step in range(interval // self.ohlc_intervals[0])
                            ],
                            "last_timestamps": [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ].index[-1]
                                for step in range(interval // self.ohlc_intervals[0])
                            ],
                            "last_train_timestamps": [
                                self.ohlc_data[pair][interval][
                                    self.ohlc_intervals[0] * step
                                ]
                                .loc[: self.train_end]
                                .index[-1]
                                for step in range(interval // self.ohlc_intervals[0])
                            ],
                            "lenghts": [
                                len(
                                    self.ohlc_data[pair][interval][
                                        self.ohlc_intervals[0] * step
                                    ]
                                )
                                for step in range(interval // self.ohlc_intervals[0])
                            ],
                            "train_steps": [
                                len(
                                    self.ohlc_data[pair][interval][
                                        self.ohlc_intervals[0] * step
                                    ].loc[: self.train_end]
                                )
                                for step in range(interval // self.ohlc_intervals[0])
                            ],
                        }
                        for pair in self.pairs
                    },
                }
                for interval in self.ohlc_intervals
            },
        }

        return metadata

    def save_arrays(self, filename):
        if len(filename.split(".")) == 1:
            filename += ".npz"
        if len(filename.split(".")) == 2 and filename.split(".")[-1] != "npz":
            raise ValueError("file name must end with .npz")
        np.savez(
            filename,
            **{str(k): v for k, v in zip(self.ohlc_intervals, self.get_arrays())},
        )

    def save_compressed_arrays(self, filename):
        if len(filename.split(".")) == 1:
            filename += ".npz"
        if len(filename.split(".")) == 2 and filename.split(".")[-1] != "npz":
            raise ValueError("file name must end with .npz")
        np.savez_compressed(
            filename,
            **{str(k): v for k, v in zip(self.ohlc_intervals, self.get_arrays())},
        )

    def save(self, filename):
        if len(filename.split(".")) == 1:
            filename += ".pkl"
        if len(filename.split(".")) == 2 and filename.split(".")[-1] != "pkl":
            raise ValueError("file name must end with .pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def save_metadata(self, filename, to_pickle=True):
        if to_pickle:
            if len(filename.split(".")) == 1:
                filename += ".pkl"
            if len(filename.split(".")) == 2 and filename.split(".")[-1] != "pkl":
                raise ValueError("file name must end with .pkl")
            with open(filename, "wb") as file:
                pickle.dump(self.get_metadata(), file)
        else:
            if len(filename.split(".")) == 1:
                filename += ".json"
            if len(filename.split(".")) == 2 and filename.split(".")[-1] != "json":
                raise ValueError("file name must end with .json")
            with open(filename, "w") as fp:
                json.dump(self.get_metadata(), fp, cls=JSONEncoder)


class DataGenerator:
    def __init__(
        self,
        dataset=None,
        metadata=None,
        arrays=None,
        y_funcs=None,
        val_test_split=0.7,
        y_standarize=True,
    ):
        if dataset is not None:
            with open(dataset, "rb") as file:
                dataset = pickle.load(file)
                self.metadata = dataset.get_metadata()
                self.x = dataset.get_arrays()
        else:
            if metadata is None or arrays is None:
                raise ValueError(
                    "metadata and arrays cant be None if dataset is not provided"
                )
            with open(metadata, "rb") as file:
                if metadata.endswith(".pkl"):
                    self.metadata = pickle.load(file)
                else:
                    self.metadata = json.load(file)
            arrays = np.load(arrays)
            self.x = [
                arrays[str(interval)] for interval in self.metadata["ohlc_intervals"]
            ]

        self.val_test_split = val_test_split
        self.y_standarize = y_standarize

        self.y_funcs = y_funcs
        self.max_len_y = 0
        self.y_names = []
        self.y = []
        self.y_mean = None
        self.y_std = None

        # get most important metadata for better readability
        self.ohlc_intervals = self.metadata["ohlc_intervals"]
        self.pairs = self.metadata["pairs"]
        self.columns_order = self.metadata["dataframe_stats"][self.ohlc_intervals[0]][
            "columns_order"
        ]
        self.dataset_seq_len = [
            self.metadata["dataset_seq_len"][interval]
            for interval in self.ohlc_intervals
        ]
        self.n_others = len(self.ohlc_intervals[1:])
        self.others_len = np.array([x.shape[1] for x in self.x[1:]])

        self.base_pairs_len = np.array(
            [
                self.metadata["dataframe_stats"][self.ohlc_intervals[0]]["pairs"][pair][
                    "lenghts"
                ][0]
                for pair in self.pairs
            ]
        )
        self.base_pairs_len = np.cumsum(self.base_pairs_len)
        self.base_pairs_len = np.insert(self.base_pairs_len, 0, 0)

        self.others_pairs_len = []
        for interval in self.ohlc_intervals[1:]:
            arr = np.array(
                [
                    self.metadata["dataframe_stats"][interval]["pairs"][pair]["lenghts"]
                    for pair in self.pairs
                ]
            )
            arr = np.cumsum(arr, axis=0)
            arr = arr.T
            arr = np.insert(arr, 0, 0, axis=1)
            self.others_pairs_len.append(arr)

        # calculate posible offsets values, usefull during sampling
        self.freq = []
        self.offsets = []
        for interval in self.ohlc_intervals[1:]:
            self.freq.append(interval // self.ohlc_intervals[0])
            offsets = [
                self.ohlc_intervals[0] * step
                for step in range(interval // self.ohlc_intervals[0])
            ]
            offsets = sorted(offsets, reverse=True)[:-1]
            offsets.insert(0, 0)
            self.offsets.append(np.array(offsets))
        self.freq = np.array(self.freq)

        # We can define any objective Y.
        # If Y is not given, the value of 'close' will be predicted for the next 5 time steps
        if self.y_funcs is None:
            column = "close"
            self.y_funcs = [
                (
                    f"pct_change_{i}",
                    partial(
                        pct_change,
                        base_column=column,
                        future_column=column,
                        future_steps=i,
                    ),
                )
                for i in range(1, 10)
            ]
        # the maximum number of steps forward for the predicted values.
        # It determines how many last samples will be dropped,
        # we have no information for them about the future values of the predicted feature.
        for name, func in self.y_funcs:
            array, max_len_y = func(
                self.x[0], self.columns_order, seq_len=self.dataset_seq_len[0]
            )
            self.y.append(array)
            self.y_names.append(name)
            self.max_len_y = max(self.max_len_y, max_len_y)
        self.y = np.stack(self.y, axis=1)

        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
        ) = self._calculate_indices()

        if y_standarize:
            self.y_mean = self.y[self.train_indices].mean(axis=0)
            self.y_std = self.y[self.train_indices].std(axis=0)
            self.y = (self.y - self.y_mean) / self.y_std

    def _calculate_indices(self):
        # Arrays are concatenated, we need to carefully choose indexes from which the data will be sampled
        # so as not to select fragments containing data from 2 pairs.
        # numpy.lib.stride_tricks.as_strided is also used to sampling with memory efficient manner
        base_pair_lenghts = [
            v["lenghts"][0]
            for k, v in self.metadata["dataframe_stats"][self.ohlc_intervals[0]][
                "pairs"
            ].items()
        ]
        base_train_lenghts = [
            v["train_steps"][0]
            for k, v in self.metadata["dataframe_stats"][self.ohlc_intervals[0]][
                "pairs"
            ].items()
        ]
        seq_len = self.dataset_seq_len[0]
        n_pairs = len(base_pair_lenghts)

        forbidden_indices = []
        for i in range(n_pairs):
            # remove indexes with mixed pairs
            start_pos = sum(base_pair_lenghts[: i + 1]) - seq_len + 1
            forbidden = list(range(start_pos, start_pos + seq_len - 1))
            forbidden_indices.extend(forbidden)
            # remove indices that does not contain calculable y features
            forbidden = list(range(start_pos, start_pos - self.max_len_y - 1, -1))
            forbidden_indices.extend(forbidden)

        train_indices = []
        val_indices = []
        test_indices = []
        for i in range(n_pairs):
            start_pos = sum(base_pair_lenghts[:i])
            last_pos = base_pair_lenghts[i]
            train_indices.extend(
                list(range(start_pos, start_pos + base_train_lenghts[i] - seq_len + 1))
            )

            val_test_indices = list(
                range(
                    start_pos + base_train_lenghts[i] - seq_len + 1,
                    start_pos + last_pos,
                )
            )
            val_test_indices = [
                idx for idx in val_test_indices if idx not in forbidden_indices
            ]
            split_idx = int(len(val_test_indices) * self.val_test_split)
            val_indices.extend(val_test_indices[:split_idx])
            test_indices.extend(val_test_indices[split_idx:])
        train_indices = [idx for idx in train_indices if idx not in forbidden_indices]

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

    def get_generator(
        self,
        indices,
        batch_size=32,
        replace=False,
        feature_last=True,
        price_normalize=None,
        price_normalize_column=None,
        seed=None,
    ):

        base_seq_len = self.dataset_seq_len[0]
        base_window = np.lib.stride_tricks.sliding_window_view(
            self.x[0], base_seq_len, axis=0
        )
        other_windows = [
            np.lib.stride_tricks.sliding_window_view(
                x.reshape((x.shape[0] * x.shape[1], x.shape[2])), seq_len, axis=0
            )
            for x, seq_len in zip(self.x[1:], self.dataset_seq_len[1:])
        ]
        if feature_last:
            base_window = np.moveaxis(base_window, -1, -2)
            other_windows = [np.moveaxis(window, -1, -2) for window in other_windows]

        if price_normalize:
            if price_normalize not in [
                "pct_change",
                "max_pct_change",
                "last_pct_change",
            ]:
                raise ValueError(
                    "price_normalize can be: 'pct_change', 'max_pct_change' or 'last_pct_change'"
                )

            price_normalization_factors = np.array(
                [
                    self.metadata["dataframe_stats"][interval]["price_std"][
                        price_normalize_column
                    ][price_normalize]
                    for interval in self.ohlc_intervals
                ]
            )
            price_indices = [
                np.where(
                    np.isin(
                        self.metadata["dataframe_stats"][interval]["columns_order"],
                        self.metadata["price_col"] + self.metadata["pair_col"],
                    )
                )[0]
                for interval in self.ohlc_intervals
            ]

            price_normalize_indices = [
                np.where(
                    self.metadata["dataframe_stats"][interval]["columns_order"]
                    == price_normalize_column
                )[0][0]
                for interval in self.ohlc_intervals
            ]

        base_freqs = []
        base_values = []
        for i, pair in enumerate(self.pairs):
            base_first_timestamp = self.metadata["dataframe_stats"][
                self.ohlc_intervals[0]
            ]["pairs"][pair]["first_timestamps"][0]
            base_freq = base_first_timestamp.freq.nanos
            base_value = base_first_timestamp.value
            base_freqs.append(base_freq)
            base_values.append(base_value)
        base_freqs = np.array(base_freqs)
        base_values = np.array(base_values)

        other_freqs = []
        result = []
        for i, pair in enumerate(self.pairs):
            result.append([])
            other_freqs.append([])
            for interval in self.ohlc_intervals[1:]:
                timestamps = self.metadata["dataframe_stats"][interval]["pairs"][pair][
                    "first_timestamps"
                ]
                interval_freq = timestamps[0].freq.nanos
                other_freqs[i].append(interval_freq)
                timestamps = [t.value for t in timestamps]
                result[i].append(
                    [(t - base_freqs[i]) % interval_freq for t in timestamps]
                    + [
                        np.nan
                        for _ in range(
                            int(self.ohlc_intervals[-1] / self.ohlc_intervals[0])
                            - len(timestamps)
                        )
                    ]
                )
        result = np.array(result)
        other_freqs = np.array(other_freqs)

        while True:
            if seed is not None:
                np.random.seed(seed)
            selected = np.random.choice(indices, batch_size, replace=replace)
            x_batch = [base_window[selected]]

            ind = np.searchsorted(self.base_pairs_len, selected + 1) - 1
            reductor = self.base_pairs_len[ind]
            idx = selected - reductor

            mod = np.mod(
                np.repeat(
                    ((idx + self.dataset_seq_len[0]) * base_freqs[ind])[np.newaxis, :],
                    self.n_others,
                    axis=0,
                ).T,
                other_freqs[ind],
            )
            x = np.repeat(mod[:, :, np.newaxis], result.shape[-1], axis=-1)
            offset_id = np.where(result[ind] == x)[-1].reshape((-1, self.n_others)).T

            for i in range(self.n_others):
                offset = offset_id[i]
                new_idx = (
                    offset * self.others_len[i]
                    + idx // self.freq[i]
                    + self.others_pairs_len[i][offset, ind]
                )
                x_batch.append(other_windows[i][new_idx])

            if price_normalize:
                for i, x in enumerate(x_batch):
                    x[:, :, price_indices[i]] = (
                        (
                            x[:, :, price_indices[i]]
                            - x[
                                :,
                                -1,
                                price_normalize_indices[i],
                                np.newaxis,
                                np.newaxis,
                            ]
                        )
                        / x[:, -1, price_normalize_indices[i], np.newaxis, np.newaxis]
                        / price_normalization_factors[i]
                    )

            for x in x_batch:
                x[np.isnan(x)] = 0

            yield x_batch, self.y[selected]

    def get_train_generator(
        self,
        batch_size=32,
        replace=False,
        feature_last=True,
        price_normalize=None,
        price_normalize_column=None,
        seed=None,
    ):
        return self.get_generator(
            self.train_indices,
            batch_size=batch_size,
            replace=replace,
            feature_last=feature_last,
            price_normalize=price_normalize,
            price_normalize_column=price_normalize_column,
            seed=seed,
        )

    def get_val_generator(
        self,
        batch_size=32,
        replace=False,
        feature_last=True,
        price_normalize=None,
        price_normalize_column=None,
        seed=None,
    ):
        return self.get_generator(
            self.val_indices,
            batch_size=batch_size,
            replace=replace,
            feature_last=feature_last,
            price_normalize=price_normalize,
            price_normalize_column=price_normalize_column,
            seed=seed,
        )

    def get_test_generator(
        self,
        batch_size=32,
        replace=False,
        feature_last=True,
        price_normalize=None,
        price_normalize_column=None,
        seed=None,
    ):
        return self.get_generator(
            self.test_indices,
            batch_size=batch_size,
            replace=replace,
            feature_last=feature_last,
            price_normalize=price_normalize,
            price_normalize_column=price_normalize_column,
            seed=seed,
        )


class GeneratorTester:
    def __init__(self, data_generator):
        self.data_generator = data_generator
        self.metadata = data_generator.metadata
        self.ohlc_intervals = self.metadata["ohlc_intervals"]

    def test_ohlc_data(self, n=1000, batch_size=32):
        intervals = self.ohlc_intervals
        close_loc = [
            np.where(self.metadata["dataframe_stats"][i]["columns_order"] == "close")[
                0
            ][0]
            for i in self.ohlc_intervals
        ]
        open_loc = [
            np.where(self.metadata["dataframe_stats"][i]["columns_order"] == "open")[0][
                0
            ]
            for i in self.ohlc_intervals
        ]
        high_loc = [
            np.where(self.metadata["dataframe_stats"][i]["columns_order"] == "high")[0][
                0
            ]
            for i in self.ohlc_intervals
        ]
        low_loc = [
            np.where(self.metadata["dataframe_stats"][i]["columns_order"] == "low")[0][
                0
            ]
            for i in self.ohlc_intervals
        ]

        batch_time, test_time = 0, 0

        gen = self.data_generator.get_train_generator(
            price_normalize=False,
            batch_size=batch_size,
        )

        errors = {
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
        }

        for _ in range(n):
            tmp = monotonic()
            x_batch, y_batch = next(gen)
            batch_time += monotonic() - tmp
            tmp = monotonic()
            for i, interval_a in enumerate(intervals):
                for j, interval_b in enumerate(intervals[i + 1 :]):
                    close_a = x_batch[i][:, :, close_loc[i]]
                    close_b = x_batch[j + i + 1][:, :, close_loc[j + i + 1]]
                    errors["close"] += self.check_close(
                        interval_a, interval_b, close_a, close_b
                    )

                    open_a = x_batch[i][:, :, open_loc[i]]
                    open_b = x_batch[j + i + 1][:, :, open_loc[j + i + 1]]
                    errors["open"] += self.check_open(
                        interval_a, interval_b, open_a, open_b
                    )

                    high_a = x_batch[i][:, :, high_loc[i]]
                    high_b = x_batch[j + i + 1][:, :, high_loc[j + i + 1]]
                    errors["high"] += self.check_high(
                        interval_a, interval_b, high_a, high_b
                    )

                    low_a = x_batch[i][:, :, low_loc[i]]
                    low_b = x_batch[j + i + 1][:, :, low_loc[j + i + 1]]
                    errors["low"] += self.check_low(
                        interval_a, interval_b, low_a, low_b
                    )
            test_time += monotonic() - tmp

        print(
            f"batch size: {batch_size}, n tests: {n}, batch generation took: {batch_time}, testing took: {test_time}"
        )
        return errors

    @staticmethod
    def check_close(interval_a, interval_b, array_a, array_b, verbose=False):
        intervals_lcm = lcm(interval_a, interval_b)
        list_len = array_a.shape[1]

        counter_error = 0
        for idx in range(array_a.shape[0]):
            a = array_a[idx]
            b = array_b[idx]

            idxs_a = [
                -int(i * intervals_lcm / interval_a) - 1
                for i in range(list_len // (intervals_lcm // interval_a))
            ]
            idxs_b = [
                -int(i * intervals_lcm / interval_b) - 1
                for i in range(list_len // (intervals_lcm // interval_a))
            ]
            selected_a = a[idxs_a]
            selected_b = b[idxs_b]

            for i in range(len(idxs_a)):
                if selected_a[i] != selected_b[i]:
                    counter_error += 1
                    if verbose:
                        print("close error")
                        print(interval_a, interval_b, idx, a, b, selected_a, selected_b)
        return counter_error

    @staticmethod
    def check_open(interval_a, interval_b, array_a, array_b, verbose=False):
        intervals_lcm = lcm(interval_a, interval_b)
        list_len = array_a.shape[1]

        counter_error = 0
        for idx in range(array_a.shape[0]):
            a = array_a[idx]
            b = array_b[idx]

            offset_a = intervals_lcm // interval_a - 1
            offset_b = intervals_lcm // interval_b - 1
            idxs_a = [
                -int(i * intervals_lcm / interval_a) - 1 - offset_a
                for i in range(list_len // (intervals_lcm // interval_a))
            ]
            idxs_b = [
                -int(i * intervals_lcm / interval_b) - 1 - offset_b
                for i in range(list_len // (intervals_lcm // interval_a))
            ]
            selected_a = a[idxs_a]
            selected_b = b[idxs_b]

            for i in range(len(idxs_a)):
                if selected_a[i] != selected_b[i]:
                    counter_error += 1
                    if verbose:
                        print("open error")
                        print(interval_a, interval_b, idx, a, b, selected_a, selected_b)
        return counter_error

    @staticmethod
    def check_high(interval_a, interval_b, array_a, array_b, verbose=False):
        intervals_lcm = lcm(interval_a, interval_b)
        list_len = array_a.shape[1]

        for idx in range(array_a.shape[0]):
            a = array_a[idx]
            b = array_b[idx]

        idxs_a = [
            (
                list_len - int((i - 1) * intervals_lcm / interval_a),
                list_len - int(i * intervals_lcm / interval_a),
            )
            for i in range(1, list_len // (intervals_lcm // interval_a))
        ]
        idxs_b = [
            (
                list_len - int((i - 1) * intervals_lcm / interval_b),
                list_len - int(i * intervals_lcm / interval_b),
            )
            for i in range(1, list_len // (intervals_lcm // interval_a))
        ]
        selected_a = [list(a[start:stop]) for stop, start in idxs_a[::-1]]
        selected_b = [list(b[start:stop]) for stop, start in idxs_b[::-1]]

        counter_error = 0
        for i in range(len(idxs_a)):
            if max(selected_a[i]) != max(selected_b[i]):
                counter_error += 1
                if verbose:
                    print("high error")
                    print(interval_a, interval_b, idx, a, b, selected_a, selected_b)
        return counter_error

    @staticmethod
    def check_low(interval_a, interval_b, array_a, array_b, verbose=False):
        intervals_lcm = lcm(interval_a, interval_b)
        list_len = array_a.shape[1]

        for idx in range(array_a.shape[0]):
            a = array_a[idx]
            b = array_b[idx]

        idxs_a = [
            (
                list_len - int((i - 1) * intervals_lcm / interval_a),
                list_len - int(i * intervals_lcm / interval_a),
            )
            for i in range(1, list_len // (intervals_lcm // interval_a))
        ]
        idxs_b = [
            (
                list_len - int((i - 1) * intervals_lcm / interval_b),
                list_len - int(i * intervals_lcm / interval_b),
            )
            for i in range(1, list_len // (intervals_lcm // interval_a))
        ]
        selected_a = [list(a[start:stop]) for stop, start in idxs_a[::-1]]
        selected_b = [list(b[start:stop]) for stop, start in idxs_b[::-1]]

        counter_error = 0
        for i in range(len(idxs_a)):
            if min(selected_a[i]) != min(selected_b[i]):
                counter_error += 1
                if verbose:
                    print("low error")
                    print(interval_a, interval_b, idx, a, b, selected_a, selected_b)
        return counter_error