import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

file_path = 'LCTGen.csv'
output_file = 'LCTGen_fleiss_kappa_result.txt'

df = pd.read_csv(file_path)

mapping = {
    'Totally match': 0,
    'Mostly match': 1,
    'Partially match': 2,
    'Mostly not match': 3,
    'Totally not match': 4
}

for col in df.columns[1:]:
    df[col] = df[col].map(mapping)

num_cases = df.shape[0]
num_categories = len(mapping)
category_counts = np.zeros((num_cases, num_categories), dtype=int)

for i, row in df.iterrows():
    for response in row[1:]:
        category_counts[i, response] += 1

# Fleiss' Kappa
kappa = fleiss_kappa(category_counts)

with open(output_file, 'w') as file:
    file.write("Fleiss' Kappa coeff: {:.4f}\n".format(kappa))

print(f"Fleiss' Kappa saved in {output_file}!")
