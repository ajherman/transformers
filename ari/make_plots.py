import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV files
df1 = pd.read_csv('original/metrics.csv')
df2 = pd.read_csv('relu/metrics.csv')

# Extract the second columns
epoch1 = df1.iloc[:, 0]
epoch2 = df2.iloc[:, 0]

logppl1 = df1.iloc[:, 1]
logppl2 = df2.iloc[:, 1]

ppl1 = df1.iloc[:, 2]
ppl2 = df2.iloc[:, 2]

# Plot the data
plt.plot(epoch1, logppl1, label='GeLU')
plt.plot(epoch2, logppl2, label='ReLU')

# Add a horizontal dashed line at y=3.5
plt.axhline(y=3.5, color='r', linestyle='--', label='Baseline')

# Set the labels for the axes
plt.xlabel('epoch')
plt.ylabel('Log PPL')
# Add a legend
plt.legend()
# Show the plot
plt.show()

# Set the labels for the axes
plt.xlabel('epoch')
plt.ylabel('Log PPL')

# Add a legend
plt.legend()

# Show the plot
plt.show()