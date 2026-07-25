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
        # (mean=0, std=1), helpful for many ML algorithms,
        # especially those that rely on distance metrics.
        self.scaler = StandardScaler()

        # Get the features.csv column names.
        self.features_cols = pd.read_csv("data/features.csv").columns.tolist()

        # We are interested in predicting weekly sales, so we set the target
        # column accordingly.
        self.target_col = "Weekly_Sales"

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal components using ISO calendar rules.
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

        x = df[self.features_cols].values
        y = df[self.target_col].values

        x_scaled = self.scaler.transform(x)
        return x_scaled, y
