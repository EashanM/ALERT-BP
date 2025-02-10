import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class MixedTypeImputer:
    """
    Custom imputer for mixed data types using MICE methodology.
    Handles both numerical and categorical variables.
    """

    def __init__(self, max_iter=10, random_state=42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        self.categorical_columns = None
        self.numerical_columns = None

    def _prepare_categorical_data(self, X, fit=True):
        """
        Encode categorical variables using LabelEncoder
        """
        X_encoded = X.copy()

        # Convert categorical columns to string type first
        for col in self.categorical_columns:
            X_encoded[col] = X_encoded[col].astype(str)

        if fit:
            self.label_encoders = {}
            for col in self.categorical_columns:
                le = LabelEncoder()
                # Get non-null values and add 'missing' category
                valid_values = X_encoded[col][X_encoded[col].notna()].unique()
                all_values = np.append(valid_values, ['missing'])
                le.fit(all_values)
                self.label_encoders[col] = le

                # Transform non-null values
                non_null_mask = X_encoded[col].notna()
                X_encoded.loc[non_null_mask, col] = le.transform(
                    X_encoded.loc[non_null_mask, col]
                )
                X_encoded[col] = X_encoded[col].astype(float)
        else:
            for col in self.categorical_columns:
                # Handle potential new categories by setting them to 'missing'
                non_null_mask = X_encoded[col].notna()
                valid_values = X_encoded.loc[non_null_mask, col].copy()
                X_encoded.loc[non_null_mask, col] = self.label_encoders[col].transform(
                    valid_values.astype(str)
                )
                X_encoded[col] = X_encoded[col].astype(float)

        return X_encoded

    def _reverse_categorical_encoding(self, X):
        """
        Reverse the label encoding for categorical variables
        """
        X_decoded = X.copy()

        for col in self.categorical_columns:
            X_decoded[col] = self.label_encoders[col].inverse_transform(
                X_decoded[col].round().astype(int)
            )
            # Replace 'missing' with np.nan
            X_decoded.loc[X_decoded[col] == 'missing', col] = np.nan

        return X_decoded

    def fit(self, X):
        """
        Fit the imputer on the training data
        """
        # Convert categorical dtypes to string to avoid issues with categorical dtype
        X = X.copy()
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].astype(str)

        # Identify categorical and numerical columns
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns

        # Prepare categorical data
        if len(self.categorical_columns) > 0:
            X_encoded = self._prepare_categorical_data(X, fit=True)

            # Initialize categorical imputer
            self.categorical_imputer = IterativeImputer(
                estimator=RandomForestClassifier(n_estimators=100),
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            self.categorical_imputer.fit(X_encoded[self.categorical_columns])

        # Initialize numerical imputer
        if len(self.numerical_columns) > 0:
            self.numerical_imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100),
                random_state=self.random_state,
                max_iter=self.max_iter
            )
            self.numerical_imputer.fit(X[self.numerical_columns])

        return self

    def transform(self, X):
        """
        Transform the data by imputing missing values
        """
        X_imputed = X.copy()

        # Convert categorical dtypes to string
        for col in X_imputed.select_dtypes(include=['category']).columns:
            X_imputed[col] = X_imputed[col].astype(str)

        # Impute categorical variables
        if len(self.categorical_columns) > 0:
            X_encoded = self._prepare_categorical_data(X_imputed, fit=False)
            categorical_imputed = self.categorical_imputer.transform(
                X_encoded[self.categorical_columns]
            )
            X_encoded[self.categorical_columns] = categorical_imputed
            X_imputed = self._reverse_categorical_encoding(X_encoded)

        # Impute numerical variables
        if len(self.numerical_columns) > 0:
            numerical_imputed = self.numerical_imputer.transform(
                X_imputed[self.numerical_columns]
            )
            X_imputed[self.numerical_columns] = numerical_imputed

        return X_imputed

    def fit_transform(self, X):
        """
        Fit the imputer and transform the data
        """
        return self.fit(X).transform(X)


def create_mixed_sample_data(n_samples=1000):
    """
    Create sample dataset with both numerical and categorical variables
    """
    np.random.seed(42)

    # Create numerical features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = x1 * 0.5 + np.random.normal(0, 0.5, n_samples)

    # Create categorical features
    cat1 = np.random.choice(['A', 'B', 'C'], size=n_samples)
    cat2 = np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], size=n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'category1': cat1,
        'category2': cat2
    })

    # Introduce missing values randomly
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.2
        df.loc[mask, col] = np.nan

    return df


def test_imputation():
    """
    Test the mixed-type imputer
    """
    # Create sample data
    df = create_mixed_sample_data()

    print("Original data sample:")
    print(df.head())
    print("\nMissing values summary:")
    print(df.isnull().sum())

    # Perform imputation
    imputer = MixedTypeImputer()
    df_imputed = imputer.fit_transform(df)

    print("\nImputed data sample:")
    print(df_imputed.head())
    print("\nMissing values after imputation:")
    print(df_imputed.isnull().sum())

    # Value distribution comparison
    for col in df.columns:
        print(f"\nValue distribution for {col}:")
        print("Original (non-missing):")
        print(df[col].value_counts(normalize=True).head())
        print("\nImputed:")
        print(df_imputed[col].value_counts(normalize=True).head())


if __name__ == "__main__":
    test_imputation()