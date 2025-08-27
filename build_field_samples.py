# ✅ Updated build_field_samples.py
import os
import pandas as pd
import json

CSV_DIR = "filtered_csvdata"
IMG_DIR = "image_data/BMW"

# Read CSVs
basic = pd.read_csv(os.path.join(CSV_DIR, "basic_bmw_2008_2018.csv"))
price = pd.read_csv(os.path.join(CSV_DIR, "price_bmw_2008_2018.csv"))
ad = pd.read_csv(os.path.join(CSV_DIR, "ad_bmw_2008_2018.csv"))
trim = pd.read_csv(os.path.join(CSV_DIR, "trim_bmw_2008_2018.csv"))

# ✅ Ensure Fuel_type is not dropped
def safe_drop(df, columns):
    return df.drop(columns=[col for col in columns if col in df.columns and col not in ["Fuel_type"]], errors="ignore")

# Drop duplicate columns (excluding Fuel_type)
basic = safe_drop(basic, ["Genmodel", "Maker"])
ad = safe_drop(ad, ["Genmodel", "Maker"])
trim = safe_drop(trim, ["Genmodel", "Maker"])

# Deduplicate to avoid join explosion
basic = basic.drop_duplicates(subset=["Genmodel_ID"])
ad = ad.drop_duplicates(subset=["Genmodel_ID"])
trim = trim.drop_duplicates(subset=["Genmodel_ID"])

# Merge dataframes
merged_df = price.merge(basic, on="Genmodel_ID", how="left") \
                 .merge(ad, on="Genmodel_ID", how="left", suffixes=("", "_ad")) \
                 .merge(trim, on="Genmodel_ID", how="left", suffixes=("", "_trim"))

# Prefer 'Fuel_type' from 'ad' (drop if duplicate from trim)
if "Fuel_type_trim" in merged_df.columns:
    merged_df.drop(columns=["Fuel_type_trim"], inplace=True)

# Ensure 'Fuel_type' exists
if "Fuel_type" not in merged_df.columns:
    raise ValueError("❌ Fuel_type not found in merged_df after merge.")

# Drop rows where critical fields are missing
merged_df = merged_df.dropna(subset=["Entry_price", "Fuel_type"])
merged_df["Genmodel_ID"] = merged_df["Genmodel_ID"].astype(str)

# Scan images
image_map = {}
for root, _, files in os.walk(IMG_DIR):
    for fname in files:
        if fname.endswith(".jpg") and "$$" in fname:
            try:
                parts = fname.split("$$")
                genmodel_id = parts[4]
                full_path = os.path.join(root, fname).replace("\\", "/")
                image_map[genmodel_id] = full_path
            except IndexError:
                continue

# Build sample list
samples = []
for _, row in merged_df.iterrows():
    genmodel_id = str(row["Genmodel_ID"])
    if genmodel_id in image_map:
        samples.append({
            "Genmodel_ID": genmodel_id,
            "Fuel_type": row["Fuel_type"],
            "Image_path": image_map[genmodel_id]
        })

# Save
with open("field_samples_final.json", "w") as f:
    json.dump(samples, f, indent=2)

print(f"✅ field_samples_final.json created with {len(samples)} valid samples.")
