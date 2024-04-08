import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset from the specified file path
file_path = "C:/Users/teelh05/Downloads/baseball.xlsx"
data = pd.read_excel(file_path)

# Extracting relevant columns from the dataset
runs_scored = data['Runs Scored']
runs_allowed = data['Runs Allowed']
wins = data['Wins']
team_batting_avg = data['Team Batting Average']
obp = data['OBP']
slg = data['SLG']

# Function to perform linear regression and visualize the results
def perform_linear_regression_and_plot(x, y, x_label, y_label, title):
    # Reshaping x to fit the requirements of scikit-learn
    x_reshaped = x.values.reshape(-1, 1)
    
    # Create a linear regression model
    model = LinearRegression()
    # Fit the model to the data
    model.fit(x_reshaped, y)
    # Predict y values using the model
    y_pred = model.predict(x_reshaped)
    # Calculate R-squared value
    r_squared = r2_score(y, y_pred)
    
    # Plotting
    plt.scatter(x, y, color='blue', label='Data')  # Scatter plot of the data
    plt.plot(x, y_pred, color='red', label='Regression Line')  # Regression line
    plt.xlabel(x_label)  # X-axis label
    plt.ylabel(y_label)  # Y-axis label
    plt.title(title)  # Chart title
    plt.text(max(x) * 0.75, min(y) * 0.75, f'R-squared: {r_squared:.2f}')  # Display R-squared value
    plt.legend()  # Show legend
    plt.show()  # Display the plot

# Perform linear regression and plot for the first case
perform_linear_regression_and_plot(runs_scored - runs_allowed, wins, 'Runs Difference', 'Wins', 'Wins vs Runs Difference')

# Perform linear regression and plot for the second case
perform_linear_regression_and_plot(team_batting_avg, runs_scored - runs_allowed, 'Team Batting Average', 'Runs Difference', 'Runs Difference vs Team Batting Average')

# Perform multiple regression for the third case
X = data[['OBP', 'SLG']]  # Independent variables
y = runs_scored - runs_allowed  # Dependent variable
model = LinearRegression().fit(X, y)  # Fit the multiple regression model

# Print out regression statistics
print("Multiple Regression Statistics:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R-squared: {model.score(X, y)}")

# Explanation of regression statistics:
# - Coefficients represent the weights assigned to each independent variable in the regression equation.
# - Intercept represents the value of the dependent variable when all independent variables are zero.
# - R-squared is a measure of how well the independent variables explain the variability of the dependent variable.

print('Go Brewers!')