

import pandas as pd
import glob
import os

# ==============================
# 1. Path to dataset folder
# ==============================
data_path = "../../data/security/*.csv"

# ==============================
# 2. Get all CSV files in folder
# ==============================
files = glob.glob(data_path)

print(f"Number of files found: {len(files)}")




# ==============================
# 3. Read each file and store in a list
# ==============================
dfs = []

for file in files:
    print(f"Loading: {file}")

    # Read each CSV file
    df = pd.read_csv(file)

    # Append to list
    dfs.append(df)
# ==============================
# 4. Merge all datasets together
# ==============================
final_df = pd.concat(dfs, ignore_index=True)

# ==============================
# 5. Check result
# ==============================
print(final_df.shape)
print(final_df.head())


# Save merged dataset for next steps
final_df.to_csv("../../data/security/merged_data.csv", index=False)



