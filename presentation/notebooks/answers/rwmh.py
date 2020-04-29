import matplotlib.pylab as plt
import numpy as np
from scipy.stats import beta
from tqdm import tqdm


plt.style.use('seaborn-darkgrid')


def posterior(theta, n_flips=10, n_heads=4):

    # Hyperparameters
    a, b = 2, 2

    # Uniform prior distribution
    prior = beta.pdf(theta, a=a, b=b)

    # Compute the likelihood
    likelihood = theta ** n_heads * (1 - theta) ** (n_flips - n_heads)

    # Compute the posterior distribution (up to a constant)
    posterior = prior * likelihood

    return posterior


def proposal(theta_current):
    return theta_current + np.random.normal(0.0, 0.2)


# initialization
thetas = [0.5]
n_samples = 1000
accepted_proposals = 0

# sample
for i in tqdm(range(n_samples)):

    theta_current = thetas[-1]
    theta_proposal = proposal(theta_current)
    alpha = posterior(theta_proposal) / posterior(theta_current)
    u = np.random.uniform()
    if alpha > u:
        theta_new = theta_proposal
        accepted_proposals += 1
    else:
        theta_new = theta_current

    thetas.append(theta_new)

print(f"Acceptance rate={accepted_proposals / n_samples:.2f}")

_, ax = plt.subplots(figsize=(8, 5))
ax.hist(thetas, bins=30)
plt.show()

_, ax = plt.subplots(figsize=(8, 5))
ax.plot(thetas)
plt.show()


