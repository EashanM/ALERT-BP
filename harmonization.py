import pandas as pd


def harmonize_datasets(df1, df2):
    """
    Harmonize two datasets based on common variables and standardize their structures
    """

    # Dictionary for column name mappings
    column_mappings = {

        # Maternal characteristics
        'mborig': 'maternal_origin',
        'mbledu': 'maternal_education',
        'mbincom': 'household_income',
        's1hhincome_new': 'household_income',
        'Nethnic_mom': 'maternal_ethnicity',
        'mbleduyrs': 'maternal_education_years',

        # Medical history
        'anygdm_BiB': 'gestational_diabetes',
        'anygdm': 'gestational_diabetes',
        'gdm_report': 'gdm_birth_report',
        'gestHT': 'gestational_hypertension',
        'mblhdis': 'maternal_heart_disease',
        'mblcvd': 'maternal_cvd',
        'mblpreht': 'preexisting_hypertension',
        'priordiabp': 'preexisting_diabetes',

        # Paternal conditions

        'f1ycvd': 'paternal_cvd',

        'fblcvd': 'paternal_cvd',

        'f1ydiab': 'paternal_diabetes',

        'fbldiab': 'paternal_diabetes',

        'f1yht': 'paternal_hypertension',

        'fblht': 'paternal_hypertension',

        # Smoking

        'mblncig': 'maternal_cigarettes_per_day',

        'mblsmkexp': 'maternal_smoke_exposure',

        'msmkhist': 'maternal_smoking_history',

        # Child characteristics

        'csex': 'child_sex',
        'cbthweight': 'birth_weight',
        'gestage': 'gestational_age',
        'gestwgtgain': 'gestational_weight_gain',

        # Breastfeeding

        'everbfed': 'ever_breastfed',
        'stopbfed': 'breastfeeding_duration'

    }

    # Standardize categorical variables

    categorical_mappings = {

        'child_sex': {
            1: 'Male',
            2: 'Female',
            3: 'Ambiguous',
            4: 'Unknown'

        },

        'maternal_ethnicity': {

            1: 'Hispanic',

            2: 'European',

            3: 'South Asian',

            4: 'Arab',

            5: 'East/South-East Asian',

            6: 'African',

            7: 'Indigenous',

            8: 'Mixed',

            9: 'Other',

            10: 'Unknown'

        },

        'maternal_smoking_history': {

            0: 'Never smoked',

            1: 'Quit before pregnancy',

            2: 'Quit during pregnancy',

            3: 'Currently smoking'

        },

        'household_income': {

            1: '<$15,000',

            2: '$15,000-$29,999',

            3: '$30,000-$49,999',

            4: '≥$50,000'

        }

    }

    # Binary variable standardization

    binary_variables = [

        'gestational_diabetes', 'gdm_birth_report', 'gestational_hypertension',

        'maternal_heart_disease', 'maternal_cvd', 'preexisting_hypertension',

        'preexisting_diabetes', 'paternal_cvd', 'paternal_diabetes',

        'paternal_hypertension', 'ever_breastfed'

    ]

    def standardize_binary(x):

        """Standardize binary variables to 0/1 format"""

        if isinstance(x, str) and x.upper() == 'Y':

            return 1

        elif isinstance(x, str) and x.upper() == 'N':

            return 0

        elif x in [0, 1]:

            return x

        else:

            return None

    def harmonize_dataframe(df):

        """Apply harmonization to a single dataframe"""

        # Create a copy to avoid modifying the original

        df_harmonized = df.copy()

        # Rename columns based on mapping

        df_harmonized = df_harmonized.rename(columns=column_mappings)

        # Standardize categorical variables

        for col, mapping in categorical_mappings.items():

            if col in df_harmonized.columns:
                df_harmonized[col] = df_harmonized[col].map(mapping)

        # Standardize binary variables

        for col in binary_variables:

            if col in df_harmonized.columns:
                df_harmonized[col] = df_harmonized[col].apply(standardize_binary)

        # Handle special cases for income

        if 'mbincom' in df.columns:
            income_map = {

                1: 1,  # $0-14,999

                2: 1,  # $15,000-19,999

                3: 2,  # $20,000-29,999

                4: 3,  # $30,000-39,999

                5: 3,  # $40,000-49,999

                6: 4,  # $50,000+

                7: 4,

                8: 4,

                9: 4

            }

            df_harmonized['household_income'] = df_harmonized['household_income'].map(income_map)

        return df_harmonized

    # Apply harmonization to both datasets

    df1_harmonized = harmonize_dataframe(df1)

    df2_harmonized = harmonize_dataframe(df2)

    return df1_harmonized, df2_harmonized


# Function to verify harmonization

def verify_harmonization(df1, df2):
    """

    Verify that the harmonization was successful by comparing variable distributions

    """

    common_cols = set(df1.columns).intersection(set(df2.columns))

    for col in common_cols:
        print(f"\nVariable: {col}")

        print("Dataset 1 unique values:", df1[col].unique())

        print("Dataset 2 unique values:", df2[col].unique())


# Example usage:

"""

# Load your datasets

df1 = pd.read_csv('dataset1.csv')

df2 = pd.read_csv('dataset2.csv')


# Harmonize datasets

df1_harmonized, df2_harmonized = harmonize_datasets(df1, df2)


# Verify harmonization

verify_harmonization(df1_harmonized, df2_harmonized)

"""


# Additional utility functions for data quality checks

def check_missing_values(df1, df2):
    """

    Check missing values in both datasets

    """

    missing_df1 = df1.isnull().sum()

    missing_df2 = df2.isnull().sum()

    missing_comparison = pd.DataFrame({

        'Dataset1_Missing': missing_df1,

        'Dataset1_Missing_Pct': (missing_df1 / len(df1)) * 100,

        'Dataset2_Missing': missing_df2,

        'Dataset2_Missing_Pct': (missing_df2 / len(df2)) * 100

    })

    return missing_comparison


def check_value_distributions(df1, df2):
    """

    Compare value distributions between datasets for common variables

    """

    common_cols = set(df1.columns).intersection(set(df2.columns))

    for col in common_cols:

        if df1[col].dtype in ['int64', 'float64']:

            print(f"\nNumerical Variable: {col}")

            print("\nDataset 1 Summary:")

            print(df1[col].describe())

            print("\nDataset 2 Summary:")

            print(df2[col].describe())

        else:

            print(f"\nCategorical Variable: {col}")

            print("\nDataset 1 Value Counts:")

            print(df1[col].value_counts(normalize=True))

            print("\nDataset 2 Value Counts:")

            print(df2[col].value_counts(normalize=True))

    class DatasetHarmonizer:

        """

        A class to manage the harmonization process and provide additional utilities

        """

        def __init__(self, df1, df2):

            self.df1_original = df1.copy()

            self.df2_original = df2.copy()

            self.df1_harmonized = None

            self.df2_harmonized = None

        def harmonize(self):

            """

            Execute the harmonization process

            """

            self.df1_harmonized, self.df2_harmonized = harmonize_datasets(

                self.df1_original,

                self.df2_original

            )

        def export_data(self, path1='harmonized_dataset1.csv', path2='harmonized_dataset2.csv'):

            """

            Export harmonized datasets to CSV files

            """

            if self.df1_harmonized is not None and self.df2_harmonized is not None:

                self.df1_harmonized.to_csv(path1, index=False)

                self.df2_harmonized.to_csv(path2, index=False)

            else:

                raise ValueError("Datasets have not been harmonized yet. Run harmonize() first.")

        def get_variable_summary(self):

            """

            Generate a summary of variables in both datasets

            """

            if self.df1_harmonized is None or self.df2_harmonized is None:
                raise ValueError("Datasets have not been harmonized yet. Run harmonize() first.")

            variables_df1 = set(self.df1_harmonized.columns)

            variables_df2 = set(self.df2_harmonized.columns)

            summary = {

                'common_variables': variables_df1.intersection(variables_df2),

                'unique_to_df1': variables_df1 - variables_df2,

                'unique_to_df2': variables_df2 - variables_df1

            }

            return summary

        def generate_codebook(self):

            """

            Generate a codebook for the harmonized datasets

            """

            if self.df1_harmonized is None or self.df2_harmonized is None:
                raise ValueError("Datasets have not been harmonized yet. Run harmonize() first.")

            codebook = []

            for column in set(self.df1_harmonized.columns) | set(self.df2_harmonized.columns):

                entry = {

                    'variable_name': column,

                    'present_in_df1': column in self.df1_harmonized.columns,

                    'present_in_df2': column in self.df2_harmonized.columns,

                    'data_type': str(self.df1_harmonized[column].dtype) if column in self.df1_harmonized.columns

                    else str(self.df2_harmonized[column].dtype),

                    'categories': None

                }

                # Get unique values for categorical variables

                if column in categorical_mappings:

                    entry['categories'] = categorical_mappings[column]

                elif column in binary_variables:

                    entry['categories'] = {0: 'No', 1: 'Yes'}

                codebook.append(entry)

            return pd.DataFrame(codebook)

    def main():

        """

        Main function to demonstrate usage

        """

        # Example usage

        print("To use this harmonization script:")

        print("\n1. Load your datasets