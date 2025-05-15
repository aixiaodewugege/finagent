root = None
workdir = "workdir"
tag = "processd_day_exp_stocks"
batch_size = 5

processor = dict(
    type = "Processor",
    root = root,
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path":r"C:\Users\90701\projects\data.csv",
            }
        ],
        "news": [
            {
                "type": "fmp",
                "path":r"C:\Users\90701\projects\data.csv",
            },
        ],
    },
    start_date = "2020-01-01",
    end_date = "2024-01-01",
    interval = "1d",
    if_parse_url = False,
    stocks_path = "configs/_asset_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag
)