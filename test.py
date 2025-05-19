import pandas as pd

df = pd.read_parquet(r"C:\Users\90701\projects\finagent\datasets\exp_stocks\features\AAPL.O.parquet")
print(df.head())
