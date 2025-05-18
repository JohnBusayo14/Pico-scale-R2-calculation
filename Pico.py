import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
n_no_obstacle = 54
n_obstacle = 6
total_samples = n_no_obstacle + n_obstacle

# Define feature ranges
wind_speed = np.random.uniform(2, 10, total_samples)  # 2–10 m/s
aoa = np.random.uniform(0, 25, total_samples)  # 0–25°
obstacle_height = np.concatenate([np.zeros(n_no_obstacle), np.array([1.4, 1.53, 1.7, 2.0, 2.3, 2.5])])  # 0 for no-obstacle, 1.4–2.5 m
obstacle_distance = np.random.uniform(1.9, 2.1, total_samples)  # 1.9–2.1 m to avoid multicollinearity

# Generate synthetic power output
power_output = np.zeros(total_samples)
for i in range(total_samples):
    if obstacle_height[i] == 0:  # No-obstacle
        power_output[i] = 4.5 * wind_speed[i] + 0.5 * aoa[i] + np.random.normal(0, 2)  # ~37.961 W
    else:  # Obstacle
        power_output[i] = 6.470 - (6.470 + 0.068) * (obstacle_height[i] - 1.4) / (2.5 - 1.4) + np.random.normal(0, 0.5)

# Create DataFrame
data = pd.DataFrame({
    'wind_speed': wind_speed,
    'AoA': aoa,
    'obstacle_height': obstacle_height,
    'obstacle_distance': obstacle_distance,
    'power_output': power_output
})

# Define features and target
X = data[['wind_speed', 'AoA', 'obstacle_height', 'obstacle_distance']]
y = data['power_output']

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 5-fold cross-validation
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

# Calculate p-values using statsmodels
X_train_sm = sm.add_constant(X_train)  # Add intercept
model_sm = sm.OLS(y_train, X_train_sm).fit()
p_values = model_sm.pvalues[1:]  # Exclude intercept

# Debug: Check p_values length
print(f"Number of features: {len(X.columns)}")
print(f"Number of p-values: {len(p_values)}")
print(f"p-values: {p_values}")

# Print results
print("\nMultiple Linear Regression Results:")
print(f"R² (Test Set): {r2:.4f}")
print(f"RMSE (Test Set): {rmse:.4f} W")
print(f"Cross-Validated R² (5-fold): {cv_r2:.4f}")
print("\nRegression Coefficients:")
print(f"Intercept (β₀): {model.intercept_:.4f}")
for i, col in enumerate(X.columns):
    if i < len(p_values):  # Ensure index is within p_values bounds
        print(f"{col} (β_{i+1}): {model.coef_[i]:.4f}, p-value: {p_values.iloc[i]:.4f}")
    else:
        print(f"{col} (β_{i+1}): {model.coef_[i]:.4f}, p-value: Not available")
