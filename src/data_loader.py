import pandas as pd


def load_and_merge_data(
    data_dir: str = "data", is_test: bool = False
) -> pd.DataFrame:
    """
    Loads and merges the Walmart dataset files for future sales forecasting.
    """
    # Read the CSV files into pandas DataFrames.
    target_file = "test.csv" if is_test else "train.csv"

    main_df = pd.read_csv(f"{data_dir}/{target_file}")
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    features = pd.read_csv(f"{data_dir}/features.csv")

    # Merge the dataframes for a complete view of the data.
    # This ensures that we combine sales data with store metadata and
    # external factors (like holidays and temperature) to have a
    # better picture of what drives purchasing behaviour.
    merged_df = main_df.merge(stores, on="Store", how="left")
    merged_df = merged_df.merge(
        features, on=["Store", "Date", "IsHoliday"], how="left"
    )

    if not is_test:
        # Walmart only started tracking markdowns (discounts) on 2011-11-11.
        # Before that, values were filled as NaN to denote "no discount",
        # so we fill missing values with 0 to allow proper training.
        # There are only 5 markdown columns, so this can be hard-coded.
        markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
        merged_df[markdown_cols] = merged_df[markdown_cols].fillna(0)

    # Finally, we can return the merged and cleaned dataframe
    # for further processing.
    return merged_df
