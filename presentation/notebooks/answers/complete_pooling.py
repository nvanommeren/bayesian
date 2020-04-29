import pymc3 as pm
with pm.Model() as pooled_model:

    # sample a and b
    b0 = pm.Normal('b0', mu=0, sigma=50)
    b1 = pm.Lognormal('b1', mu=0, sigma=50)

    # compute the purchase probability for each incidence
    prob = pm.Deterministic('prob', pm.invlogit(b0 - b1 * price_data['price']))
    likelihood = pm.Bernoulli('likelihood', p=prob, observed=price_data['buy'])
    posterior = pm.sample(draws=500, tune=500)

# pm.model_to_graphviz(pooled_model)
