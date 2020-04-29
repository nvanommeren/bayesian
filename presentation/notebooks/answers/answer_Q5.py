# 1 alternatively
n = len(posterior)
posterior = np.sort(posterior)
sixty_pct_quantile = int(0.6 * n)
print('Q5 atlernatively: ', posterior[sixty_pct_quantile])