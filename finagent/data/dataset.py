import os
from finagent.data import BaseDataset
from finagent.registry import DATASET
import pandas as pd

pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_rows', 100000)

@DATASET.register_module(force=True)
class Dataset(BaseDataset):
    def __init__(self,
                 root: str = None,
                 price_path: str = None,
                 news_path: str = None,
                 guidance_path: str = None,
                 sentiment_path: str = None,
                 economics_path: str = None,
                 assets_path: str = None,
                 interval: str = "day",
                 workdir: str = None,
                 tag: str = None,
                 ):
        super(Dataset, self).__init__()

        self.root = root
        self.price_path = os.path.join(root, price_path)
        self.news_path = os.path.join(root, news_path)


        self.assets_path = os.path.join(root, assets_path)
        self.interval = interval
        self.workdir = workdir
        self.tag = tag

        self.exp_path = os.path.join(self.root, self.workdir, self.tag)
        os.makedirs(self.exp_path, exist_ok=True)

        self.assets = self._init_assets()
        self.prices = self._load_prices()
        self.news = self._load_news()
        # self.guidances = self._load_guidances()
        # self.sentiments = self._load_sentiments()
        # self.economics = self._load_economics()

    def _init_assets(self):
        with open(self.assets_path) as op:
            assets = [line.strip() for line in op.readlines()]
        return assets

    def _load_prices(self):

        prices = {}

        for asset in self.assets:
            path = os.path.join(self.price_path, "{}.parquet".format(asset))
            df = pd.read_parquet(path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.sort_values(by="timestamp")
            df = df.reset_index(drop=True)

            df = df[["timestamp", "open", "high", "low", "close", "last_close", "volume",'PE_ratio', 'PE_ratio_ttm', 'PCF_ratio_ttm', 'PB_ratio', 'PS_ratio',
       'PS_ratio_ttm',"RSI", "ADX", "BB_lower", "BB_middle", "BB_upper", "VWMA", "CCI", "MACD", "MACD_signal"]]

            prices[asset] = df

        return prices

    def _load_news(self):

        news = {}

        global_id = 0

        for asset in self.assets:
            path = os.path.join(self.news_path, "{}.parquet".format(asset))
            df = pd.read_parquet(path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d"))
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = df.sort_values(by="timestamp")
            df = df.reset_index(drop=True)

            df["id"] = df.index + global_id
            df["id"] = df["id"].apply(lambda x: "{:06d}".format(x))
            global_id += len(df)

            df = df[["timestamp", "id", "type", "title", "text"]]

            news[asset] = df

        return news


if __name__ == '__main__':

    root = "workspace/RA/FinAgentPrivate"

    dataset = Dataset(
        root = root,
        price_path = "datasets/exp_stocks/price",
        news_path = "datasets/exp_stocks/news",
        guidance_path = "datasets/exp_stocks/guidance",
        sentiment_path = "datasets/exp_stocks/sentiment",
        economics_path = "datasets/exp_stocks/economic.parquet",
        interval = "1d",
        assets_path = "configs/_asset_list_/exp_assets.txt",
        workdir = os.path.join(root, "workdir"),
        tag = "exp"
    )

    selected_asset = "AAPL"

    print(len(dataset.prices[selected_asset]))
    print(len(dataset.news[selected_asset]))
    print(len(dataset.guidances[selected_asset]))
    print(len(dataset.sentiments[selected_asset]))
