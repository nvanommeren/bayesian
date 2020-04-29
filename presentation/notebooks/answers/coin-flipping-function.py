import numpy as np

# good: a naive implementation
def flip_coin_naive(number_of_flips, theta):

    outcome = []

    for _ in range(number_of_flips):

        # generate a random number between 0 and 1
        random_number = np.random.uniform()

        # store heads as 1 and tails as 0
        if random_number < theta:
            outcome.append(1)
        else:
            outcome.append(0)

    return np.asarray(outcome)

# better: a vectorized implementation
def flip_coin_vectorized(number_of_flips, theta):
    return np.where(np.random.uniform(size=number_of_flips) < theta, 1, 0)

# best: use the build-in NumPy function
# np.random.binomial(n=1, p=0.5, size=10)

print(flip_coin_naive(10, 0.5))
print(flip_coin_vectorized(10, 0.5))
print(np.random.binomial(n=1, p=0.5, size=10))