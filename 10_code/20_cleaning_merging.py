import pandas as pd
import numpy as np

# Read in the data

winners = pd.read_csv("../00_data/oscarsWinners_10_22.csv")

# Removing the brackets and quotes from the top critics column

winners["top_critics"] = winners["top_critics"].str.replace(
    r"[\"\[\]']", "", regex=True
)

# Selecting columns

data = winners[["movie", "winner", "top_critics"]]

# Loading the data without nominations

d1 = pd.read_csv("../00_data/sample_10_22_aarya.csv")
d2 = pd.read_csv("../00_data/sample_10_22_rafa.csv")
d3 = pd.read_csv("../00_data/sample_10_22_matt.csv")

# Concatenating the data

data_wo_nominations = pd.concat([d1, d2, d3])

# Adding a column for the winner

data_wo_nominations["winner"] = -1

# Removing the brackets and quotes from the top critics column

data_wo_nominations["top_critics"] = data_wo_nominations["top_critics"].str.replace(
    r"[\"\[\]']", "", regex=True
)

# Selecting columns

data_wo_nominations = data_wo_nominations[["movie", "winner", "top_critics"]]

# Concatenating the data

data = pd.concat([data, data_wo_nominations])

# Starting from 0 to 2

data["winner"] = data["winner"] + 1

# Droping rows with no top critics

data = data[data["top_critics"].isna() == False]

data = data[data["top_critics"] == ""]

data = data[~data["top_critics"].isnull()]

# Saving the data

data.to_csv("../00_data/final_data.csv", index=False)
