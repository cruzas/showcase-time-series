import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class DataPreprocessor:
    """
    A class for pre-processing the Walmart sales data.
    """

    def __init__(self):
        """
        Initializes the DataPreprocessor with encoders and scalers for
        pre-processing the Walmart sales data.
        """

        # For encoding categorical features into numerical values.
        self.encoder = OrdinalEncoder()

        # For scaling numerical features to a standard distribution
        # (mean=0, std=1), helpful for many ML algorithms, especially those
        # that rely on distance metrics. This prevents high-magnitude features
        # from obscuring lower-magnitude variables (like temperature), ensuring
        # that the model weighs all factors in a similar fashion.
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # Isolates features with continuous values for scaling.
        self.cont_cols = [
            "Year",
            "Month",
            "Week",
            "IsHoliday",
            "Type",
            "Size",
            "Temperature",
            "Fuel_Price",
        ]

        # Isolates the categorical IDs for embeddings.
        self.cat_cols = ["Store", "Dept"]

        # We are interested in predicting weekly sales, so that is
        # our target column.
        self.target_col = "Weekly_Sales"

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal components, i.e., those that refer to time in some
        way.
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["IsHoliday"] = df["IsHoliday"].astype(
            int
        )  # Convert boolean to int for ML models.

        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        """
        Fits encoders and scalers on training data and returns a scaled
        array and targets.

        Note: It's possible to do everything below with just PyTorch, but
        it would require more code and manual handling of the mean and
        standard deviation.
        """
        df = self.extract_time_features(df)

        # With fit_transform() the OrdinalEncoder takes the 'Type' column and
        # transforms it to numerical values.
        # Double brackets are used to ensure compatibility with encoders that
        # expect a 2D array.
        df[["Type"]] = self.encoder.fit_transform(df[["Type"]])

        x_cont = self.x_scaler.fit_transform(df[self.cont_cols].values)
        x_cat = df[self.cat_cols].values.astype(int)
        y_scaled = self.y_scaler.fit_transform(
            df[self.target_col].values.reshape(-1, 1)
        )

        # Flatten returns a 1D tensor.
        return (
            x_cont,
            x_cat,
            y_scaled.flatten(),
        )

    def transform(self, df: pd.DataFrame) -> tuple:
        """
        As opposed to fit_transform, this doesn't store computed values
        (mean, std) in the encoders and scalers.
        """
        df = self.extract_time_features(df)
        df[["Type"]] = self.encoder.transform(df[["Type"]])

        x_cont = self.x_scaler.transform(df[self.cont_cols].values)
        x_cat = df[self.cat_cols].values.astype(int)
        y_scaled = self.y_scaler.transform(
            df[self.target_col].values.reshape(-1, 1)
        )

        return x_cont, x_cat, y_scaled.flatten()
