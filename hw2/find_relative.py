import numpy as np
import pandas as pd

data = pd.read_csv('ML2025Spring-hw2-public/train.csv')

tested_positive_day3 = data["tested_positive_day3"].to_numpy()

corrcoef = [
	np.corrcoef(data[col].to_numpy(), tested_positive_day3)[0, 1]
	for col in data.columns
]

results = dict(zip(data.columns[:-1], corrcoef[:-1]))

high_relative_features = [
	key \
	for key, val in results.items() \
	if abs(val) > 0.91
]

print(high_relative_features)