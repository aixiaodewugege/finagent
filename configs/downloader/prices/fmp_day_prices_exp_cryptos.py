root = None
workdir = "workdir"
tag = "fmp_day_prices_exp_cryptos"
batch_size = 1

downloader = dict(
    type = "FMPDayPriceDownloader",
    root = root,
    token = None,
    start_date = "2023-12-01",
    end_date = "2025-04-12",
    interval = "1d",
    delay = 1,
    stocks_path = "configs/_stock_list_/exp_cryptos.txt",
    workdir = workdir,
    tag = tag
)