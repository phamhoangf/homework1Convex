import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

# Load data (replace with your data loading logic)
data_train = np.loadtxt("xy_train.csv", delimiter=",")
X_train = data_train[:, :2]  # Features (training data)
y_train = data_train[:, 2]  # Labels (training data)

data_test = np.loadtxt("xy_test (1).csv", delimiter=",")
X_test = data_test[:, :2]  # Features (test data)
y_test = data_test[:, 2]  # Labels (test data)

# Define function to solve SVM and calculate misclassification error
def solve_svm_and_get_error(C):
  # Define variables
  beta = cvx.Variable((2,))  # Coefficients (2 dimensions)
  beta0 = cvx.Variable()     # Intercept
  xi = cvx.Variable((len(y_train),))  # Slack variables

  # Objective function (L2 norm regularization + cost for slack variables)
  objective = cvx.Minimize(cvx.square(cvx.norm(beta)) + C * cvx.sum((xi)))

  # Constraints (non-negativity for slack, margin condition)
  constraints = [xi >= 0 for i in range(len(y_train))]
  for i in range(len(y_train)):
      constraints.append(y_train[i] * (X_train[i] @ beta + beta0) >= 1 - xi[i])

  # Solve the SVM problem
  prob = cvx.Problem(objective, constraints)
  prob.solve()

  # Extract results
  optimal_beta = beta.value
  optimal_beta0 = beta0.value

  # Predict labels on test data using the trained model
  predicted_labels = np.sign(np.dot(X_test, optimal_beta) + optimal_beta0)

  # Calculate misclassification error
  misclassified_count = sum(predicted_labels != y_test)
  error_rate = misclassified_count / len(y_test)

  return error_rate

# Investigate misclassification error for different C values
C_values = np.linspace(2**(-5), 2**5, 11)  # Logarithmic spacing for C values
error_rates = []
for C in C_values:
  error_rate = solve_svm_and_get_error(C)
  error_rates.append(error_rate)

# Plot misclassification error vs C (logarithmic scale for C)
plt.plot(C_values, error_rates)
plt.xscale("log")
plt.xlabel("Cost Parameter (C)")
plt.ylabel("Misclassification Error")
plt.title("Impact of Cost Parameter (C) on SVM Performance")
plt.grid(True)
plt.show()