import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

print("Generating data.")
n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

# Fit LinearRegression models
print("Fitting model.")
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)

# Plot result
print("Displaying result.")
fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='upper left')
plt.title('Linear regression')
plt.show()
