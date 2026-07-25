import pandas as pd


def load_and_merge_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads and merges the Walmart dataset files for future sales forecasting.
    """
    # Read the CSV files into pandas DataFrames.
    train = pd.read_csv(f"{data_dir}/train.csv")
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    features = pd.read_csv(f"{data_dir}/features.csv")

    # Merge the dataframes for a complete view of the data.
    # "how='left'" keeps all rows from the left dataframe (train) and only the
    # matching rows from the right dataframe (stores and features).
    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    # Walmart only started tracking markdowns (discounts) in 2011-11-11.
    # Before that, values were filled as NaN to denote "no discount",
    # so we fill missing values with 0 to allow proper training.
    # There are only 5 markdown columns, so this can be hard-coded.
    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # Since we cannot set missing CPI/unemployment values to 0,
    # we drop the rows with missing values in these columns.
    df = df.dropna(subset=["CPI", "Unemployment"])

    # Finally, we can return the merged and cleaned dataframe
    # for further processing.
    return df
