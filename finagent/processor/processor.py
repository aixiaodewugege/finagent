import multiprocessing
from copy import deepcopy
from datetime import datetime
from finagent.registry import PROCESSOR
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# from langchain_community.document_loaders import PlaywrightURLLoader
import backoff
import time
import json
from bs4 import BeautifulSoup   # pip install beautifulsoup4
import pandas as pd
import ast

@backoff.on_exception(backoff.expo,(Exception,), max_tries=3, max_value=10, jitter=None)
def langchain_parse_url(url):

    print(">" * 30 + "Running langchain_parse_url" + ">" * 30)
    start = time.time()
    loader = PlaywrightURLLoader(urls = [url], remove_selectors=["header", "footer"])
    data = loader.load()
    if len(data) <= 0:
        return None
    content = data[0].page_content

    if "Please enable cookies" in content:
        return None
    if "Please verify you are a human" in content:
        return None
    if "Checking if the site connection is secure" in content:
        return None

    print("url: {} | content: {}".format(url, content[:100].split("\n")))
    end = time.time()
    print(">" * 30 + "Time elapsed: {}s".format(end - start) + ">" * 30)
    print("<" * 30 + "Finish langchain_parse_url" + "<" * 30)
    return content

def my_rank(x):
   return pd.Series(x).rank(pct=True).iloc[-1]

def cal_news(df):
    df["title"] = df["title"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    df["text"] = df["text"].fillna("").str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return df



def cal_factor(df, level="day"):
    # intermediate values
    df['max_oc'] = df[["open", "close"]].max(axis=1)
    df['min_oc'] = df[["open", "close"]].min(axis=1)
    df["kmid"] = (df["close"] - df["open"]) / df["close"]
    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-12)
    df["klen"] = (df["high"] - df["low"]) / df["open"]
    df['kup'] = (df['high'] - df['max_oc']) / df['open']
    df['kup2'] = (df['high'] - df['max_oc']) / (df['high'] - df['low'] + 1e-12)
    df['klow'] = (df['min_oc'] - df['low']) / df['open']
    df['klow2'] = (df['min_oc'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df["ksft"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    df.drop(columns=['max_oc', 'min_oc'], inplace=True)

    window = [5, 10, 20, 30, 60]
    for w in window:
        df['roc_{}'.format(w)] = df['close'].shift(w) / df['close']

    for w in window:
        df['ma_{}'.format(w)] = df['close'].rolling(w).mean() / df['close']

    for w in window:
        df['std_{}'.format(w)] = df['close'].rolling(w).std() / df['close']

    for w in window:
        df['beta_{}'.format(w)] = (df['close'].shift(w) - df['close']) / (w * df['close'])

    for w in window:
        df['max_{}'.format(w)] = df['close'].rolling(w).max() / df['close']

    for w in window:
        df['min_{}'.format(w)] = df['close'].rolling(w).min() / df['close']

    for w in window:
        df['qtlu_{}'.format(w)] = df['close'].rolling(w).quantile(0.8) / df['close']

    for w in window:
        df['qtld_{}'.format(w)] = df['close'].rolling(w).quantile(0.2) / df['close']

    for w in window:
        df['rank_{}'.format(w)] = df['close'].rolling(w).apply(my_rank) / w

    for w in window:
        df['imax_{}'.format(w)] = df['high'].rolling(w).apply(np.argmax) / w

    for w in window:
        df['imin_{}'.format(w)] = df['low'].rolling(w).apply(np.argmin) / w

    for w in window:
        df['imxd_{}'.format(w)] = (df['high'].rolling(w).apply(np.argmax) - df['low'].rolling(w).apply(np.argmin)) / w

    for w in window:
        shift = df['close'].shift(w)
        min = df["low"].where(df["low"] < shift, shift)
        max = df["high"].where(df["high"] > shift, shift)
        df["rsv_{}".format(w)] = (df["close"] - min) / (max - min + 1e-12)

    df['ret1'] = df['close'].pct_change(1)
    for w in window:
        df['cntp_{}'.format(w)] = (df['ret1'].gt(0)).rolling(w).sum() / w

    for w in window:
        df['cntn_{}'.format(w)] = (df['ret1'].lt(0)).rolling(w).sum() / w

    for w in window:
        df['cntd_{}'.format(w)] = df['cntp_{}'.format(w)] - df['cntn_{}'.format(w)]

    for w in window:
        df1 = df["close"].rolling(w)
        df2 = np.log(df["volume"] + 1).rolling(w)
        df["corr_{}".format(w)] = df1.corr(pairwise = df2)

    for w in window:
        df1 = df["close"]
        df_shift1 = df1.shift(1)
        df2 = df["volume"]
        df_shift2 = df2.shift(1)
        df1 = df1 / df_shift1
        df2 = np.log(df2 / df_shift2 + 1)
        df["cord_{}".format(w)] = df1.rolling(w).corr(pairwise = df2.rolling(w))

    df['abs_ret1'] = np.abs(df['ret1'])
    df['pos_ret1'] = df['ret1']
    df['pos_ret1'][df['pos_ret1'].lt(0)] = 0

    for w in window:
        df['sump_{}'.format(w)] = df['pos_ret1'].rolling(w).sum() / (df['abs_ret1'].rolling(w).sum() + 1e-12)

    for w in window:
        df['sumn_{}'.format(w)] = 1 - df['sump_{}'.format(w)]

    for w in window:
        df['sumd_{}'.format(w)] = 2 * df['sump_{}'.format(w)] - 1

    for w in window:
        df["vma_{}".format(w)] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-12)

    for w in window:
        df["vstd_{}".format(w)] = df["volume"].rolling(w).std() / (df["volume"] + 1e-12)

    for w in window:
        shift = np.abs((df["close"] / df["close"].shift(1) - 1)) * df["volume"]
        df1 = shift.rolling(w).std()
        df2 = shift.rolling(w).mean()
        df["wvma_{}".format(w)] = df1 / (df2 + 1e-12)

    df['vchg1'] = df['volume'] - df['volume'].shift(1)
    df['abs_vchg1'] = np.abs(df['vchg1'])
    df['pos_vchg1'] = df['vchg1']
    df['pos_vchg1'][df['pos_vchg1'].lt(0)] = 0

    for w in window:
        df["vsump_{}".format(w)] = df["pos_vchg1"].rolling(w).sum() / (df["abs_vchg1"].rolling(w).sum() + 1e-12)
    for w in window:
        df["vsumn_{}".format(w)] = 1 - df["vsump_{}".format(w)]
    for w in window:
        df["vsumd_{}".format(w)] = 2 * df["vsump_{}".format(w)] - 1

    df["log_volume"] = np.log(df["volume"] + 1)

    df.drop(columns=['ret1', 'abs_ret1', 'pos_ret1', 'vchg1', 'abs_vchg1', 'pos_vchg1', 'volume'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    if level == "minute":
        df["minute"] = pd.to_datetime(df.index).minute
        df["hour"] = pd.to_datetime(df.index).hour

    df["day"] = pd.to_datetime(df.index).day
    df["weekday"] = pd.to_datetime(df.index).weekday
    df["month"] = pd.to_datetime(df.index).month

    return df

def cal_target(df):
    df['ret1'] = df['close'].pct_change(1).shift(-1)
    df['mov1'] = (df['ret1'] > 0)
    df['mov1'] = df['mov1'].astype(int)
    return df

@PROCESSOR.register_module(force=True)
class Processor():
    def __init__(self,
                 root=None,
                 path_params = None,
                 stocks_path = None,
                 start_date = None,
                 end_date = None,
                 interval="day",
                 if_parse_url = False,
                 workdir = None,
                 tag = None
                 ):
        self.root = root
        self.path_params = path_params
        self.stocks_path = os.path.join(root, stocks_path)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.if_parse_url = if_parse_url
        self.workdir = workdir
        self.tag = tag

        self.stocks = self._init_stocks()

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def _process_price_and_features(self,
                stocks = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "last_close",
            #"市盈率", "市盈率ttm", "市现率ttm", "市净率", "市销率", "市销率ttm"
            "PE_ratio", "PE_ratio_ttm", "PCF_ratio_ttm", "PB_ratio", "PS_ratio", "PS_ratio_ttm"
        ]

        for stock in tqdm(stocks):
            price = self.path_params["prices"][0]
            price_type = price["type"]
            price_path = price["path"]

            price_path = os.path.join(self.root, price_path)

            # Define column mappings based on price_type

            price_column_map = {
                "time": "timestamp",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "收盘价": "close",
                "成交量": "volume",
                "昨日收盘价": "last_close",  # Using 昨日收盘价 as adj_close for now
            }
            # Additional columns to keep from Chinese format
            additional_columns_map = {
                "市盈率": "PE_ratio", "市盈率ttm": "PE_ratio_ttm", "市现率ttm": "PCF_ratio_ttm", "市净率": "PB_ratio", "市销率": "PS_ratio", "市销率ttm": "PS_ratio_ttm"
            }
            price_column_map.update(additional_columns_map)

            

            assert os.path.exists(price_path), "Price path {} does not exist".format(price_path)
            price_df = pd.read_csv(price_path)
            

            # Check if thscode column exists and use it to filter the stock if needed
            if "thscode" in price_df.columns:
                if stock in price_df["thscode"].values:
                    price_df = price_df[price_df["thscode"] == stock]
            
            # Select and rename columns

            columns_to_select_from_csv = [col for col in price_df.columns if col in price_column_map.keys()]
            # Select only these relevant columns from the DataFrame
            price_df = price_df[columns_to_select_from_csv]
            
            # Rename columns according to mapping
            rename_dict = {k: v for k, v in price_column_map.items() if k in price_df.columns}
            price_df = price_df.rename(columns=rename_dict)
            
            # Ensure all required columns exist
            for col in price_columns:
                if col not in price_df.columns:
                    price_df[col] = 0  # Default value for missing columns


            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
            price_df = price_df[(price_df["timestamp"] >= start_date) & (price_df["timestamp"] < end_date)]

            price_df = price_df.sort_values(by="timestamp")
            price_df = price_df.drop_duplicates(subset=["timestamp"], keep="first")
            price_df = price_df.reset_index(drop=True)

            outpath = os.path.join(self.root, self.workdir, self.tag, "price")
            os.makedirs(outpath, exist_ok=True)
            price_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)

            # For features calculation, use only the standard price columns
            features_df = cal_factor(deepcopy(price_df[["timestamp"] + price_columns]), level=self.interval)
            features_df = cal_target(features_df)
            
            
            outpath = os.path.join(self.root, self.workdir, self.tag, "features")
            os.makedirs(outpath, exist_ok=True)
            features_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)


    def _process_news(self,
                stocks = None,
                start_date = None,
                end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        news_columns = [
            "title",
            "text",
            "url"
        ]

        for stock in tqdm(stocks):
            newses = self.path_params["news"]
            newses_df = []

            for news in newses:
                news_type = news["type"]
                news_path = news["path"]

                news_path = os.path.join(self.root, news_path)
                news_column_map = {
                    "time": "timestamp",
                    "新闻": "news",
                }
                # For Chinese format, we might not have all columns
                # Set defaults for missing columns
                default_values = {
                    "title": "",
                    "text": "",
                    "url": ""
                }

                if not os.path.exists(news_path):
                    print(f"News path {news_path} does not exist, skipping...")
                    continue

                news_df = pd.read_csv(news_path)
                

                if "thscode" in news_df.columns:
                    if stock in news_df["thscode"].values:
                        news_df = news_df[news_df["thscode"] == stock]
                
                # Select and rename columns
                columns_to_select = [col for col in news_df.columns if col in news_column_map.keys()]
                if len(columns_to_select) == 0:
                    print(f"No matching columns found in {news_path}, skipping...")
                    continue
                    
                news_df = news_df[columns_to_select]
                news_df = news_df.rename(columns=news_column_map)
                


                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
                news_df = news_df[(news_df["timestamp"] >= start_date) & (news_df["timestamp"] < end_date)]
                news_df = news_df.sort_values(by="timestamp")
                

                if 'news' in news_df.columns:
                    expanded_data_list = []
                    original_columns = news_df.columns.tolist()

                    for index, row in news_df.iterrows():
                        news_items_str = row.get('news') 
                        
                        if pd.notna(news_items_str) and isinstance(news_items_str, str):
                            articles = [] # 初始化为空列表
                            try:
                                # 使用 ast.literal_eval 代替 json.loads
                                parsed_data = ast.literal_eval(news_items_str)
                                
                                if isinstance(parsed_data, list):
                                    articles = parsed_data
                                elif isinstance(parsed_data, dict):
                                    articles = [parsed_data] # 如果解析出单个字典，将其放入列表中
                                else:
                                    print(f"Warning: Parsed news data for row {index} is not a list or dict. Type: {type(parsed_data)}. Content: {str(parsed_data)[:200]}")
                                    # articles 保持为空列表

                            except (ValueError, SyntaxError) as e:
                                print(f"Error parsing news_items_str with ast.literal_eval for row {index}: {e}")
                                print(f"Problematic string (first 200 chars): {news_items_str[:200]}")
                                # articles 保持为空列表，跳过此条错误数据
                            
                            for art in articles:
                                publish_time = art.get("publish_time")
                                title        = art.get("title")
                                
                                # 提取纯文本正文
                                html_content = art.get("content", "")
                                soup = BeautifulSoup(html_content, "html.parser")
                                text = soup.get_text(" ", strip=True)          # 纯文本，段落间用空格分隔
                                
                                url = art.get("url")   # 如果 JSON 里没有 url 字段，可先为 None
                                expanded_data_list.append(
                                    {"timestamp": publish_time,
                                    "title":        title,
                                    "text":         text,
                                    "url":          url}
                                )

                expanded_news_df = pd.DataFrame(expanded_data_list)
                news_df = expanded_news_df
                
                news_df["type"] = "fmp"

                news_df = news_df.reset_index(drop=True)
                news_df = cal_news(news_df)
                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))
                newses_df.append(news_df)

            # Only proceed if we have news data
            if len(newses_df) > 0:
                newses_df = pd.concat(newses_df)

                if self.if_parse_url and "url" in newses_df.columns and not newses_df["url"].isnull().all():
                    urls = newses_df["url"].values
                    max_process = 10
                    pool = multiprocessing.Pool(processes=max_process)
                    contents = pool.map(langchain_parse_url, urls)
                    pool.close()
                    pool.join()
                    newses_df["content"] = contents

                newses_df = newses_df.sort_values(by="timestamp")
                newses_df = newses_df.drop_duplicates(subset=["timestamp", "title"], keep="first")
                newses_df = newses_df.reset_index(drop=True)
                
                # Ensure all required columns exist
                for col in ["timestamp", "type", "title", "text", "url"]:
                    if col not in newses_df.columns:
                        newses_df[col] = ""
                    
                newses_df = newses_df[["timestamp", "type", "title", "text", "url"]]

                outpath = os.path.join(self.root, self.workdir, self.tag, "news")
                os.makedirs(outpath, exist_ok=True)
                newses_df.to_parquet(os.path.join(outpath, "{}.parquet".format(stock)), index=False)
            else:
                print(f"No news data found for {stock}, skipping...")


    def process(self,
                stocks = None,
                start_date = None,
                end_date = None):

        print(">" * 30 + "Running price and features..." + ">" * 30)
        self._process_price_and_features(stocks=stocks, start_date=start_date, end_date=end_date)
        print("<" * 30 + "Finish price and features..." + "<" * 30)

        print(">" * 30 + "Running news..." + ">" * 30)
        self._process_news(stocks=stocks, start_date=start_date, end_date=end_date)
        print("<" * 30 + "Finish news..." + "<" * 30)
