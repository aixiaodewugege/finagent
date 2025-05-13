root = None
workdir = "workdir"
tag = "processd_day_exp_cryptos"
batch_size = 1

processor = dict(
    type = "Processor",
    root = root,
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path":"workdir/fmp_day_prices_exp_cryptos",
            }
        ],
        "news": [
            {
                "type": "fmp",
                "path":"workdir/fmp_news_exp_cryptos",
            }
        ]
    },
    start_date = "2024-03-12",
    end_date = "2025-04-12",
    interval = "minute",
    if_parse_url = False,
    stocks_path = "configs/_stock_list_/exp_cryptos.txt",
    workdir = workdir,
    tag = tag
)