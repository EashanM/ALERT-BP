import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class MICEImputer:
    def __init__(self, max_iter=10, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = None
        self.categorical_columns = None
        self.label_encoders = {}

    def fit_transform(self, X):
        # Identify categorical columns (including float-based categories)
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_columns += [col for col in X.select_dtypes(include=['float64']).columns
                                     if X[col].nunique() / len(
                X) < 0.05]  # Assuming columns with less than 5% unique values are categorical

        # Create label encoders for categorical columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Handle NaN values before encoding
            non_nan_values = X[col].dropna()
            le.fit(non_nan_values.astype(str))
            self.label_encoders[col] = le

            # Transform non-NaN values, leave NaNs as is
            X[col] = X[col].apply(lambda x: le.transform([str(x)])[0] if pd.notnull(x) else x)

        # Create the IterativeImputer
        self.imputer = IterativeImputer(
            estimator=self._get_estimator(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy='most_frequent'
        )

        # Fit and transform the data
        imputed_data = self.imputer.fit_transform(X)

        # Convert back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=X.columns)

        # Reverse label encoding for categorical columns
        for col in self.categorical_columns:
            non_nan_mask = imputed_df[col].notnull()
            imputed_df.loc[non_nan_mask, col] = self.label_encoders[col].inverse_transform(
                imputed_df.loc[non_nan_mask, col].astype(int))

            # Convert back to float if original column was float
            if X[col].dtype == 'float64':
                imputed_df[col] = pd.to_numeric(imputed_df[col], errors='coerce')

        return imputed_df

    def _get_estimator(self):
        return RandomForestRegressor(random_state=self.random_state)

# Example usage
def main():
    # Create a sample dataset with missing values
    data = pd.DataFrame({
        'age': [25, 30, np.nan, 40, 35],
        'income': [50000, np.nan, 75000, 90000, 60000],
        'education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor'],
        'city': ['New York', 'London', 'Paris', np.nan, 'Tokyo']
    })
    X5 = pd.read_csv("../data/X5yr.csv")
    columns_to_encode = ['Nethnic_mom', 'csex', 'everbfed', 'gdm_report', 'mblcvd', 'mblhdis', 'msmkhist', 'priordiabp']
    X5[columns_to_encode] = X5[columns_to_encode].astype('category')

    data = X5

    data = data.replace('NaN', np.nan)

    print("Original data:")
    print(data)

    # Initialize and use the MICEImputer

    mice_imputer = MICEImputer(max_iter=5, random_state=42)
    imputed_data = mice_imputer.fit_transform(data)

    print("\nImputed data:")
    print(imputed_data)

    imputed_data.to_csv("../data/imputationTest.csv", index=False)

if __name__ == "__main__":
    main()
