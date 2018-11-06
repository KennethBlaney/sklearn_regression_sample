import numpy as np
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

print("Generating Data.")
n = 100                                                             # number of data points
x = np.arange(n)                                                    # x values
random_seed = check_random_state(0)
y = random_seed.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))   # y values

# Fit IsotonicRegression models
print("Fitting model.")
ir = IsotonicRegression()
y_ = ir.fit_transform(x, y)

# Plot result
print("Displaying result.")
fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'b.-', markersize=12)
plt.legend(('Data', 'Isotonic Fit'), loc='upper left')
plt.title('Isotonic regression')
plt.show()
