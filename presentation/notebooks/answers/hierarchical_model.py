import pymc3 as pm

n_customers = price_data['customer_id'].nunique()
ids = price_data['customer_id'].values

with pm.Model() as hierarchical_model:

    # Prior distributions
    mu_b0 = pm.Normal('mu_b0', mu=0, sd=100)
    mu_b1 = pm.Normal('mu_b1', mu=0, sd=100)

    sigma_b0 = pm.InverseGamma('sigma_b0', alpha=3, beta=100)
    sigma_b1 = pm.InverseGamma('sigma_b1', alpha=3, beta=100)

    # sample a and b
    b0 = pm.Normal('b0', mu=mu_b0, sigma=sigma_b0, shape=n_customers)
    b1 = pm.Lognormal('b1', mu=mu_b1, sigma=sigma_b1, shape=n_customers)

    # compute the utility
    utility = b0[ids] - b1[ids] * price_data['price']
    likelihood = pm.Bernoulli('likelihood', logit_p=utility, observed=price_data['buy'])

    posterior = pm.sample(draws=500, tune=500)
