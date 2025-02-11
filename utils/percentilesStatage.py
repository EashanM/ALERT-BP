from utils.imputation import *
from utils.bpPercentile import *
import pandas as pd
import re

statage = pd.read_csv('../data/statage.csv')

statage = statage[statage['Agemos'] < 61]
statage = statage.iloc[:, :5]
statage_male = statage[statage['Sex'] == 1]
statage_female = statage[statage['Sex'] == 2]


def heightZ(X, L, M, S):
    if L == 0:
        z = np.log(X / M) / S
    else:
        z = (((X / M) ** L) - 1) / (L * S)
    return z

def findHeightZ(data, growth_chart = statage_male):
    height = data["height"]

    target_value = data["age"]*12

    if data["csex"] == "2":
        growth_chart = statage_female

    closest_index = (growth_chart['Agemos'] - target_value).abs().idxmin()
    L = growth_chart.loc[closest_index, "L"]
    M = growth_chart.loc[closest_index, "M"]
    S =  growth_chart.loc[closest_index, "S"]

    return heightZ(height, L, M, S)

def expectedSBP(data):
    age = data["age"]
    height_z = data["Height Z-Score"]

    if data["csex"] == "1":
        sex = "boy"
    else:
        sex = "girl"

    return calculate_expected_bp(age, height_z, sex)

def zscoreSBP(data):
    observed_sbp = data["SBP"]
    expected_sbp = data["Expected SBP"]

    if data["csex"] == "1":
        sex = "boy"
    else:
        sex = "girl"

    return calculate_bp_zscore(observed_sbp, expected_sbp, sex, measurement_type='sbp')

def percentileSBP(data):
    return calculate_bp_percentile(data["SBP Z Score"])

def expectedDBP(data):
    age = data["age"]
    height_z = data["Height Z-Score"]

    if data["csex"] == "1":
        sex = "boy"
    else:
        sex = "girl"

    return calculate_expected_bp(age, height_z, sex, measurement_type='dbp')

def zscoreDBP(data):
    observed_sbp = data["DBP"]
    expected_sbp = data["Expected DBP"]

    if data["csex"] == "1":
        sex = "boy"
    else:
        sex = "girl"

    return calculate_bp_zscore(observed_sbp, expected_sbp, sex, measurement_type='dbp')

def percentileDBP(data):
    return calculate_bp_percentile(data["DBP Z Score"])