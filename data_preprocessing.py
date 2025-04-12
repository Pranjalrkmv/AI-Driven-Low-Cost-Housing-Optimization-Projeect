# Import libraries
import pandas as pd       # For data manipulation
import numpy as np        # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns     # For advanced visualizations
from sklearn.preprocessing import MinMaxScaler  # For scaling numerical data
import pandas as pd

# Creating a sample dataset
data = pd.DataFrame({
    'material': ['Brick', 'Cement', 'Steel', 'Wood', 'Brick', 'Steel', 'Wood', 'Cement'],
    'Cost': [50, 100, 200, 75, 55, 210, 80, 105],  # Cost per unit
    'Durability': [7, 8, 10, 6, 7, 10, 5, 8],      # Durability score (out of 10)
    'Climate Suitability': [8, 9, 6, 7, 8, 6, 7, 9] # Climate rating (out of 10)
})

# Save this dataset as a CSV file (optional)
data.to_csv('housing_data.csv', index=False)

# Display the dataset
print("Sample Dataset:")
print(data)
# Fill missing numerical values with the column mean
data['Cost'].fillna(data['Cost'].mean(), inplace=True)
data['Durability'].fillna(data['Durability'].mean(), inplace=True)

# Fill missing categorical values with the most common value (mode)
data['material'].fillna(data['material'].mode()[0], inplace=True)

# Check if all missing values are handled
print("\nMissing values after handling:")
print(data.isnull().sum())
# Convert categorical 'Material' column into numerical values using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['material'])

# Display the transformed dataset
print("\nDataset after encoding:")
print(data_encoded.head())
# Bar plot for average cost of each material type
plt.figure(figsize=(10, 6))
sns.barplot(x='material', y='Cost', data=data)
plt.title('Average Cost of Different materials')
plt.xlabel('material Type')
plt.ylabel('Cost')
plt.xticks(rotation=45)
plt.show()
# Compute correlation matrix
correlation_matrix = data_encoded.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Histogram for material costs
plt.figure(figsize=(8, 5))
sns.histplot(data['Cost'], bins=20, kde=True, color='blue')
plt.title('Distribution of material Costs')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.show()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Select columns to scale
numeric_columns = ['Cost', 'Durability', 'Climate Suitability']

# Apply scaling to the numeric columns
data_scaled = data_encoded.copy()
data_scaled[numeric_columns] = scaler.fit_transform(data_encoded[numeric_columns])

# Display the scaled data
print("\nDataset after scaling:")
print(data_scaled.head())

# Save the cleaned and scaled dataset to a new CSV file
data_scaled.to_csv('cleaned_housing_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_housing_data.csv'")

# Load the cleaned dataset
data = pd.read_csv('cleaned_housing_data.csv')

# Display column names
print("Columns in the dataset:", data.columns)
