# Step 1: Data Processing 
import pandas as pandas

df=pandas.read_csv('Project_1_Data.csv')

#Step 2: Data Visualization

print("Statistical Analysis")
print(df[['X', 'Y', 'Z']].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['X'], bins=20, color='red',edgecolor='white')
plt.title('X-Coordinate Distribution')

# Plot for Y coordinate
plt.subplot(1, 3, 2)
plt.hist(df['Y'], bins=20, color='green',edgecolor='white')
plt.title('Y-Coordinate Distribution')

# Plot for Z coordinate
plt.subplot(1, 3, 3)
plt.hist(df['Z'], bins=20, color='blue',edgecolor='white')
plt.title('Z-Coordinate Distribution')

plt.tight_layout()
plt.show()

#Step 3: Correlation Matrix

import seaborn as sns

correlation_matrix = df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)

plt.title('Correlation Matrix')

plt.show()

#Step 4: Classification Model Development/Engineering

