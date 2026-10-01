import pandas as pd

# Raw full-catalog scrape -> cleaned dataset for model training.
# The older files (items.csv / items_full.csv and their cleaned versions)
# are kept as they are, so notebooks 01-03 still reproduce their results.
INPUT_PATH = "data/items_all.csv"
OUTPUT_PATH = "data/item_clean_all.csv"

# Columns that identify the same property re-posted under a different id
# (found in 03_eda: ~10% of listings were such duplicates).
DUPLICATE_COLS = ["price", "rooms", "area", "floor", "floors", "location", "hasRepair"]


def report(step, df):
    """Print how many rows are left after each step, so every filter is visible."""
    print(f"{step:<45} {len(df):>6} rows")


df = pd.read_csv(INPUT_PATH)
report("raw", df)

# The scrape is sorted by "last bumped": a listing bumped during the run can
# appear on two pages. Same id = same listing, keep one.
df = df.drop_duplicates(subset="id")
report("after dropping repeated ids", df)

# No rooms = land or commercial property, not an apartment.
df = df[df["rooms"].notna()]
report("after dropping listings without rooms", df)

# Very low prices are data errors or rentals mixed into the sale catalog.
df = df[df["price"] >= 5000].copy()
report("after dropping price < 5000", df)

# Price per m2 below 300 AZN is not a realistic sale price in this market.
df["price_per_m2"] = df["price"] / df["area"]
df = df[df["price_per_m2"] >= 300]
report("after dropping price per m2 < 300", df)

df = df[df["location"].notna()]
report("after dropping missing location", df)

df["rooms"] = df["rooms"].astype(int)
df["price_per_m2"] = df["price_per_m2"].round(2)

# Same property re-posted by sellers under different ids. If copies end up
# in both train and test, the test score is too optimistic (data leakage).
df = df.drop_duplicates(subset=DUPLICATE_COLS)
report("after dropping re-posted duplicates", df)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print("saved to", OUTPUT_PATH)

print()
print(df["city"].value_counts().head(10))