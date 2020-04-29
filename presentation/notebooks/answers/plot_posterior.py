from matplotlib import pyplot as plt
import numpy as np
from scipy.special import binom
import seaborn as sns

# Plotting settings
sns.set()


def plot_posterior(n_flips, n_heads):

    # We will plot the posterior for theta = 0.00, 0.01, ..., 0.99, 1.00
    theta = np.linspace(0, 1, 101)

    # Uniform prior distribution
    prior_distribution = np.ones(len(theta))

    # Compute the likelihood
    likelihood = theta ** n_heads * (1 - theta) ** (n_flips - n_heads)

    # Posterior distribution
    posterior = likelihood * prior_distribution

    # Normalize the posterior distribution (for comparability)
    posterior /= np.max(posterior)

    _, ax = plt.subplots(figsize=(6,3))
    ax.plot(theta, posterior)
    ax.set_xlabel("Theta", size=15)
    ax.set_ylabel("Posterior value", size=15)
    plt.show()


plot_posterior(100, 50)


from ipywidgets import interact
interact(plot_posterior, n_flips=np.arange(5, 50, 5), n_heads=np.arange(5, 50, 5))
