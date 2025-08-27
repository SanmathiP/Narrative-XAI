# build_merged_field_metadata.py

import os
import pandas as pd

CSV_DIR = "filtered_csvdata"
IMG_DIR = "image_data/BMW"
OUTPUT_PATH = "outputs/merged_field_metadata.csv"

# Load data
price = pd.read_csv(os.path.join(CSV_DIR, "price_bmw_2008_2018.csv"))
basic = pd.read_csv(os.path.join(CSV_DIR, "basic_bmw_2008_2018.csv"))
ad = pd.read_csv(os.path.join(CSV_DIR, "ad_bmw_2008_2018.csv"))
trim = pd.read_csv(os.path.join(CSV_DIR, "trim_bmw_2008_2018.csv"))

# Deduplicate pre-merge
basic = basic.drop_duplicates(subset=["Genmodel_ID"])
ad = ad.drop_duplicates(subset=["Genmodel_ID"])
trim = trim.drop_duplicates(subset=["Genmodel_ID"])

# Merge everything
merged = price.merge(basic, on="Genmodel_ID", how="left") \
              .merge(ad, on="Genmodel_ID", how="left", suffixes=("", "_ad")) \
              .merge(trim, on="Genmodel_ID", how="left", suffixes=("", "_trim"))

# Prefer Fuel_type from ad, drop trim version if present
if "Fuel_type_trim" in merged.columns:
    merged.drop(columns=["Fuel_type_trim"], inplace=True)

# Drop if Fuel_type or Entry_price is missing
merged = merged.dropna(subset=["Entry_price", "Fuel_type"])

# Filter Genmodel_IDs that have images
all_imgs = []
for root, _, files in os.walk(IMG_DIR):
    for f in files:
        if f.endswith(".jpg") and "$$" in f:
            try:
                parts = f.split("$$")
                gen_id = parts[4]
                all_imgs.append(gen_id)
            except:
                pass

merged = merged[merged["Genmodel_ID"].astype(str).isin(all_imgs)]

# Drop duplicates and save essential columns
merged = merged.drop_duplicates(subset=["Genmodel_ID"])
final = merged[["Genmodel_ID", "Fuel_type", "Genmodel"]].dropna()

# Save
os.makedirs("outputs", exist_ok=True)
final.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved: {OUTPUT_PATH} with {len(final)} rows")
