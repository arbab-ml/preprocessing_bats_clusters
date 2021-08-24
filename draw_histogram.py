import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
# Generate some data for this demonstration.
input_file="/home/rabi/Documents/Thesis/preprocessing and spectrograms generation/SM4BAT-WWNP-190218-190306_all_stats.csv"
input_data=pd.read_csv(input_file)

data = input_data["length"]

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
# plt.hist(data, bins=5, density=True, alpha=0.6, color='b')
# plt.hist(data, bins=5, stat='probability')
sns.displot(data,bins=5, stat='probability')

# Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
#title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
title="Histogram of the Length of Audio Files (5 Bins)"
plt.title(title)
plt.xlabel("Length of Audio File")
plt.ylabel("Percentage of Occurances")
plt.show()