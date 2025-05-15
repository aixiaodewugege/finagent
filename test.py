import pandas as pd

df = pd.read_parquet(r"C:\Users\90701\projects\finagent\workdir\processd_day_exp_stocks\news\AAPL.O.parquet")
print(df.head())
