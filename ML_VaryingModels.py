import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
# Step 1: Simulate Data
# ---------------------------------------------
np.random.seed(123)

n_periods = 4
obs_per_period = 100
N = n_periods * obs_per_period

periods = np.repeat([1, 2, 3, 4], obs_per_period)

# Generate predictors X1 ... X10 (normally distributed)
X = np.random.randn(N, 10)
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, 11)])
df['Period'] = periods

# True coefficients (time-varying for X1-X5)
beta0 = [2.0, 2.5, 3.0, 3.5]
beta_X1 = [1.0, 1.5, 2.0, 2.5]
beta_X2 = [0.5, 0.3, 0.1, -0.1]
beta_X3 = [2.0, 1.8, 1.6, 1.4]
beta_X4 = [-0.5, 0.0, 0.5, 1.0]
beta_X5 = [1.0, 0.8, 0.6, 0.4]

# Time-invariant coefficients for X6-X10
beta_const = [0.5, -0.2, 0.3, -1.0, 0.8]

Y = np.zeros(N)
for i in range(N):
    p = df['Period'].iloc[i]
    p_idx = p - 1
    y_val = (beta0[p_idx]
             + beta_X1[p_idx] * df['X1'].iloc[i]
             + beta_X2[p_idx] * df['X2'].iloc[i]
             + beta_X3[p_idx] * df['X3'].iloc[i]
             + beta_X4[p_idx] * df['X4'].iloc[i]
             + beta_X5[p_idx] * df['X5'].iloc[i]
             + beta_const[0] * df['X6'].iloc[i]
             + beta_const[1] * df['X7'].iloc[i]
             + beta_const[2] * df['X8'].iloc[i]
             + beta_const[3] * df['X9'].iloc[i]
             + beta_const[4] * df['X10'].iloc[i])
    # Add noise
    y_val += np.random.randn() * 0.5
    Y[i] = y_val

df['Y'] = Y
df['Period'] = df['Period'].astype('category')

# ---------------------------------------------
# Step 2: Split Data into Training (Periods 1-3) and Test (Period 4)
# ---------------------------------------------
train_df = df[df['Period'].isin([1, 2, 3])].copy()
test_df = df[df['Period'] == 4].copy()

X_train = train_df.drop(columns='Y')
y_train = train_df['Y']

X_test = test_df.drop(columns='Y')
y_test = test_df['Y']

# ---------------------------------------------
# Step 3: Feature Engineering for Varying Coefficients
# ---------------------------------------------
# We will one-hot encode the Period and create interaction terms between Period and X1-X5.
ohe = OneHotEncoder(drop='first', sparse_output=False)
ohe.fit(X_train[['Period']])  # Fit on training data only (Periods 1-3)

period_encoded_train = ohe.transform(X_train[['Period']])
period_encoded_test = ohe.transform(X_test[['Period']])

X_train_no_period = X_train.drop(columns=['Period'])
X_test_no_period = X_test.drop(columns=['Period'])

predictors_varying = [f'X{i}' for i in range(1, 6)]  # X1-X5 vary by period
predictors_fixed = [f'X{i}' for i in range(6, 11)]  # X6-X10 do not


def create_features(X_no_period, period_encoded):
    X_interactions = []
    # Add period dummies
    X_interactions.append(period_encoded)
    # Add fixed predictors
    X_fixed = X_no_period[predictors_fixed].to_numpy()
    X_interactions.append(X_fixed)
    # Interactions for varying predictors (X1-X5)
    for var_pred in predictors_varying:
        var_col = X_no_period[var_pred].values.reshape(-1, 1)
        interacted = var_col * period_encoded
        X_interactions.append(interacted)
    # Baseline (reference) period predictors X1-X5
    X_baseline = X_no_period[predictors_varying].to_numpy()
    X_interactions.append(X_baseline)

    return np.hstack(X_interactions)


X_train_final = create_features(X_train_no_period, period_encoded_train)
X_test_final = create_features(X_test_no_period, period_encoded_test)

# ---------------------------------------------
# Step 4: Train Model on Periods 1-3
# ---------------------------------------------
model = LinearRegression()
model.fit(X_train_final, y_train)

# ---------------------------------------------
# Step 5: Predict on Period 4
# ---------------------------------------------
y_pred = model.predict(X_test_final)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE (Period 4): {mse:.4f}")
print(f"Test R² (Period 4): {r2:.4f}")

# ---------------------------------------------
# Interpretation
# ---------------------------------------------
# We trained the model only on the first three periods, and then used it to predict Y in the fourth period.
# The feature engineering approach ensures that the model can have different coefficients for X1-X5 in each period.
#
# By examining the model coefficients, you can see how it's learned period-specific slopes.
coefs = model.coef_
print("Number of coefficients:", len(coefs))
print("Coefficients array:", coefs)

