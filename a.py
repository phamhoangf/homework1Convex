import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

# Load data (replace with your data loading logic)
data = np.loadtxt("xy_train.csv", delimiter=",")
X = data[:, :2]  # Features
y = data[:, 2]  # Labels

# Define variables
beta = cvx.Variable((2,))  # Coefficients (2 dimensions)
beta0 = cvx.Variable()     # Intercept
xi = cvx.Variable((len(y),))  # Slack variables

# Objective function (L2 norm regularization + cost for slack variables)
objective = cvx.Minimize(cvx.square(cvx.norm(beta)) + cvx.sum(xi))

# Constraints (non-negativity for slack, margin condition)
constraints = [xi >= 0 for i in range(len(y))]
for i in range(len(y)):
    constraints.append(y[i] * (X[i] @ beta + beta0) >= 1 - xi[i])

# Solve the SVM problem with C=1
prob = cvx.Problem(objective, constraints)
prob.solve()

# Extract results
optimal_beta = beta.value
optimal_beta0 = beta0.value
optimal_cost = prob.value
result = f'optimal beta: {optimal_beta}\n optimal beta0: {optimal_beta0}\n optimal cost: {optimal_cost}'
print(result)



