from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import seaborn as sns

# Plotting settings
sns.set()

# The data
n_heads = 4
n_flips = 10

# Consider theta = 0.00, 0.01, ..., 0.99, 1.00
theta = np.linspace(0, 1, 101)

# Each value of theta is even likely a priori
prior_distribution = np.ones(len(theta))

# Compute the likelihood
likelihood = theta ** n_heads * (1 - theta) ** (n_flips - n_heads)

# Posterior distribution
posterior = likelihood * prior_distribution

# Normalize the posterior distribution (for comparability)
posterior /= np.max(posterior)

print('Q4:')

_, ax = plt.subplots(figsize=(6,3))
ax.plot(theta, posterior)
ax.set_xlabel("$\\theta$", size=15)
ax.set_ylabel("Posterior value", size=15)
plt.show()
print('Q5: ', np.quantile(posterior, 0.60))

# 1 alternatively
n = len(posterior)
posterior = np.sort(posterior)
sixty_pct_quantile = int(0.6 * n)
print('Q5 atlernatively: ', posterior[sixty_pct_quantile])

# 2
interval = (np.quantile(posterior, 0.025), np.quantile(posterior, 0.975))
print('Q6: ', interval)