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
##############
############'
############
my_list = [1,2,3]
import numpy as np
arr= np.array(my_list)
print(arr)

my_mat =np.array([[1,2,3],[4,5,6],[7,8,9]])
print(my_mat)


import numpy as np

# Create an array with 100 elements
my_array = np.arange(100)

# Reshape it into a 5x5 matrix
# This will raise a ValueError because the total number of elements is not 25
reshaped_array = my_array.reshape((5, 5))
