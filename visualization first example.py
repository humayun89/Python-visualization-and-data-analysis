import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate some sample data
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# Create a DataFrame
df = pd.DataFrame({'Data': data})

# Summary statistics
print("Summary Statistics:")
print(df.describe())

# Histogram plot using Seaborn (distplot for older versions)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.distplot(df['Data'], kde=True, color='skyblue', bins=30)
plt.title('Histogram of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Probability Density Function plot using SciPy
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.sort(df['Data']), y=norm.pdf(np.sort(df['Data']), np.mean(df['Data']), np.std(df['Data'])), color='red')
plt.title('Probability Density Function (PDF) of Sample Data')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()

# Boxplot using Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['Data'], color='green')
