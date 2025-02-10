import numpy as np
from scipy.stats import norm

# Constants from Table B-1 for different measurements
COEFFICIENTS = {
    'boy_sbp': {
        'intercept': 102.19768,
        'age': [1.82416, 0.12776, 0.00249, -0.00135],
        'height': [2.73157, -0.19618, -0.04659, 0.00947],
        'sd': 10.7128
    },
    'girl_sbp': {
        'intercept': 102.01027,
        'age': [1.94397, 0.00598, -0.00789, -0.00059],
        'height': [2.03526, 0.02534, -0.01884, 0.00121],
        'sd': 10.4855
    },
    'boy_dbp': {
        'intercept': 61.01217,
        'age': [0.68314, -0.09835, 0.01711, 0.00045],
        'height': [1.46993, -0.07849, -0.03144, 0.00967],
        'sd': 11.6032
    },
    'girl_dbp': {
        'intercept': 60.50510,
        'age': [1.01301, 0.01157, 0.00424, -0.00137],
        'height': [1.16641, 0.12795, -0.03869, -0.00079],
        'sd': 10.9573
    }
}

def calculate_expected_bp(age, height_z, sex, measurement_type='sbp'):
    """
    Calculate expected blood pressure based on age, height Z-score, sex, and measurement type.

    Args:
        age (float): Age in years
        height_z (float): Height Z-score
        sex (str): 'boy' or 'girl'
        measurement_type (str): 'sbp' or 'dbp'

    Returns:
        float: Expected blood pressure in mmHg
    """
    key = f"{sex}_{measurement_type}"
    if key not in COEFFICIENTS:
        raise ValueError("Invalid sex or measurement type")

    coef = COEFFICIENTS[key]

    # Calculate age polynomial terms (y-10)^j
    age_diff = age - 10
    age_terms = sum(coef['age'][j] * (age_diff)**(j+1) for j in range(4))

    # Calculate height Z-score polynomial terms
    height_terms = sum(coef['height'][k] * height_z**(k+1) for k in range(4))

    # Calculate expected BP
    expected_bp = coef['intercept'] + age_terms + height_terms

    return expected_bp

def calculate_bp_zscore(observed_bp, expected_bp, sex, measurement_type='sbp'):
    """
    Calculate blood pressure Z-score.

    Args:
        observed_bp (float): Observed blood pressure in mmHg
        expected_bp (float): Expected blood pressure in mmHg
        sex (str): 'boy' or 'girl'
        measurement_type (str): 'sbp' or 'dbp'

    Returns:
        float: Blood pressure Z-score
    """
    key = f"{sex}_{measurement_type}"
    if key not in COEFFICIENTS:
        raise ValueError("Invalid sex or measurement type")

    sd = COEFFICIENTS[key]['sd']
    return (observed_bp - expected_bp) / sd

def calculate_bp_percentile(bp_zscore):
    """
    Convert blood pressure Z-score to percentile.

    Args:
        bp_zscore (float): Blood pressure Z-score

    Returns:
        float: Blood pressure percentile (0-100)
    """
    return norm.cdf(bp_zscore) * 100

def calculate_full_bp_metrics(age, height_z, observed_bp, sex, measurement_type='sbp'):
    """
    Calculate all blood pressure metrics in one function.

    Args:
        age (float): Age in years
        height_z (float): Height Z-score
        observed_bp (float): Observed blood pressure in mmHg
        sex (str): 'boy' or 'girl'
        measurement_type (str): 'sbp' or 'dbp'

    Returns:
        dict: Dictionary containing expected BP, Z-score, and percentile
    """
    expected_bp = calculate_expected_bp(age, height_z, sex, measurement_type)
    zscore = calculate_bp_zscore(observed_bp, expected_bp, sex, measurement_type)
    percentile = calculate_bp_percentile(zscore)

    return {
        'expected_bp': round(expected_bp, 2),
        'zscore': round(zscore, 3),
        'percentile': round(percentile, 1)
    }

# Example usage
def verify_example():
    """
    Verify the implementation using the example from the document:
    12-year-old boy with height Z-score = 1.28 and SBP = 120 mmHg
    """
    results = calculate_full_bp_metrics(
        age=12,
        height_z=1.28,
        observed_bp=120,
        sex='boy',
        measurement_type='sbp'
    )

    print(f"Example verification (12-year-old boy):")
    print(f"Expected BP: {results['expected_bp']} mmHg")
    print(f"Z-score: {results['zscore']}")
    print(f"Percentile: {results['percentile']}%")
