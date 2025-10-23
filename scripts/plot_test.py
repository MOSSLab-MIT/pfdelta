import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

mu = 200
sigma = 25
n_bins = 25
data = np.random.normal(mu, sigma, size=100)

fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Cumulative distributions.
axs[1].ecdf(data, label="CDF")
# n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
#                                cumulative=True, label="Cumulative histogram")
# x = np.linspace(data.min(), data.max())
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (x - mu))**2))
# y = y.cumsum()
# y /= y[-1]


# # Complementary cumulative distributions.
# axs[1].ecdf(data, complementary=True, label="CCDF")



# # Label the figure.
# fig.suptitle("Cumulative distributions")
# for ax in axs:
#     ax.grid(True)
#     ax.legend()
#     ax.set_xlabel("Annual rainfall (mm)")
#     ax.set_ylabel("Probability of occurrence")
#     ax.label_outer()

plt.savefig('test.png')