import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import eval_metric

# Load and preprocess the data
df = pd.read_csv('train.csv')
df = df.sample(n=19000, random_state=0)

df["store"] = df["store"].astype("str")
df["item"] = df["item"].astype("str")
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Train-test split
train = df[df["date"] < "2017-01-01"]
test = df[df["date"] >= "2017-01-01"]

# Drop date and extract features if needed
train_features = train.drop(columns=["sales", "date"])
test_features = test.drop(columns=["sales", "date"])
train_target = train["sales"]
test_target = test["sales"]

# Train model
model = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=0,iterations=500,depth=9,learning_rate=0.1)
model.fit(train_features, train_target)

# Predict and evaluate
preds = model.predict(test_features)
smape_score = eval_metric(test_target.values, preds, "SMAPE")
print("SMAPE:", smape_score)

# --- Predict next 3 months (Jan–Mar 2018) ---
# Get all unique store-item combinations
store_item_pairs = df[["store", "item"]].drop_duplicates()

# Generate future dates
future_dates = pd.date_range(start="2018-01-01", end="2018-03-31")

# Create future dataframe
future_df = pd.DataFrame([(date, store, item) for date in future_dates for store, item in store_item_pairs.values],
                         columns=["date", "store", "item"])

# Convert types to match training set
future_df["store"] = future_df["store"].astype("str")
future_df["item"] = future_df["item"].astype("str")

# Drop date if not used, otherwise extract features if needed
future_features = future_df.drop(columns=["date"])

# Predict
future_df["predicted_sales"] = model.predict(future_features)

# Filter for January 2017 data
feat = df[(df["date"] >= "2017-01-01") & (df["date"] < "2017-02-01")]

# Drop target and date
feat_clean = feat.drop(columns=["sales", "date"])

# Predict
predictions = model.predict(feat_clean)

# Optional: Add predictions back to original data
feat_with_preds = feat.copy()
feat_with_preds["predicted_sales"] = predictions

# Show the first 10 results
feat_with_preds.reset_index(drop=True,inplace=True)
print(feat_with_preds[["date", "store", "item", "predicted_sales"]].head(10))


# Print or save predictions

