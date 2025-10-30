#Parsing URls using urllib from my csv


import pandas as pd
import re

df = pd.read_csv("urls.csv")

df["extracted"] = df["url"].str.extract(r"pymnts\.com/([^,]+)")[0]
df["extracted"] = df["extracted"].str.rstrip("/")
df["extracted"] = df["extracted"].str.replace("/", "-", regex=False)
df["extracted"] = df["extracted"].str.rstrip("-")
df["extracted"] = df["extracted"].str.lstrip("tag")

df["extracted"] = df["extracted"].str.split("-").str.join(" ")
print(df[["url","extracted"]])

#saving my df as a new csv file
df.to_csv("extracted_urls.csv", index=False)