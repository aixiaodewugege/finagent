import pandas as pd

df = pd.read_csv('/data/Finagent/data.csv', encoding='utf-8', engine='python', nrows=100)
print(df.head())
