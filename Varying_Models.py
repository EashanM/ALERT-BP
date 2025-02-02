import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Set random seed for reproducibility
np.random.seed(123)

# --------------------------------------------------------
# Step 1: Simulate Some Synthetic Data
# --------------------------------------------------------
# Let's say we have 4 time periods and 100 observations per period, total 400 rows.
n_periods = 4
obs_per_period = 100
N = n_periods * obs_per_period

# Create a time period variable
periods = np.repeat([1, 2, 3, 4], obs_per_period)

# Generate predictors: X1 ... X10
# We assume they are independent standard normal variables
X = np.random.randn(N, 10)
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, 11)])
df['Period'] = periods

# --------------------------------------------------------
# Step 2: Define a True Model for the Synthetic Data
# --------------------------------------------------------
# Let's define "true" coefficients.
# Intercepts for each period:
beta0 = [2.0, 2.5, 3.0, 3.5]  # increasing intercept over time

# For X1 to X5, let's have varying slopes per period.
# For simplicity, let’s create a pattern:
#   X1 slope grows by 0.5 each period starting from 1.0
#   X2 slope grows by -0.2 each period starting from 0.5
# and so forth
beta_X1 = [1.0, 1.5, 2.0, 2.5]
beta_X2 = [0.5, 0.3, 0.1, -0.1]
beta_X3 = [2.0, 1.8, 1.6, 1.4]
beta_X4 = [-0.5, 0.0, 0.5, 1.0]
beta_X5 = [1.0, 0.8, 0.6, 0.4]

# For X6 to X10, coefficients remain constant (time-invariant):
beta_const = [0.5, -0.2, 0.3, -1.0, 0.8]  # For X6 to X10 respectively

# --------------------------------------------------------
# Step 3: Generate the Response Variable Y
# --------------------------------------------------------
Y = np.zeros(N)

for i in range(N):
    p = df['Period'].iloc[i]
    # Period index (0 to 3)
    p_idx = p - 1

    # Calculate intercept and time-varying portion
    y_val = (beta0[p_idx] +
             beta_X1[p_idx] * df['X1'].iloc[i] +
             beta_X2[p_idx] * df['X2'].iloc[i] +
             beta_X3[p_idx] * df['X3'].iloc[i] +
             beta_X4[p_idx] * df['X4'].iloc[i] +
             beta_X5[p_idx] * df['X5'].iloc[i])

    # Add time-invariant coefficients
    y_val += (beta_const[0] * df['X6'].iloc[i] +
              beta_const[1] * df['X7'].iloc[i] +
              beta_const[2] * df['X8'].iloc[i] +
              beta_const[3] * df['X9'].iloc[i] +
              beta_const[4] * df['X10'].iloc[i])

    # Add some noise
    y_val += np.random.randn() * 0.5
    Y[i] = y_val

df['Y'] = Y

# Convert Period to a categorical factor
df['Period'] = df['Period'].astype('category')

# --------------------------------------------------------
# Step 4: Fit a Varying Coefficients Model
# --------------------------------------------------------
# We want to allow X1 through X5 to vary by period.
# We can do this by interacting these predictors with Period.
#
# Model formula:
# Y ~ Period + X6 + X7 + X8 + X9 + X10 + (X1 + X2 + X3 + X4 + X5)*Period
#
# This will estimate separate slopes for X1...X5 for each period.
# For X6...X10, we assume one slope across all periods.

formula = ("Y ~ Period + X6 + X7 + X8 + X9 + X10 + "
           "(X1 + X2 + X3 + X4 + X5):Period")

model = smf.ols(formula, data=df).fit()
print(model.summary())

# --------------------------------------------------------
# Step 5: Interpretation
# --------------------------------------------------------
# In the summary, you will see estimates for:
# - Intercepts for each period (via Period dummies).
# - Coefficients for X1 to X5 that differ by period (interaction terms).
# - Constant coefficients for X6 to X10.
#
# The estimated coefficients should be close to the "true" values we used,
# especially with a relatively large sample. Of course, due to random noise,
# they won't be exact, but this demonstrates the approach.