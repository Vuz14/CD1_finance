import pandas as pd
import numpy as np

# Generate sample data for testing
np.random.seed(42)

# Create sample dataset
n_samples = 1000
n_features = 20

# Features
X = np.random.randn(n_samples, n_features)
feature_names = [f'feature_{i}' for i in range(n_features)]

# Target (binary classification - default/not default)
y = np.random.binomial(1, 0.3, n_samples)

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['company_id'] = np.random.randint(1, 11, n_samples)
df['year'] = np.random.choice([2018, 2019, 2020, 2021, 2022], n_samples)

# Save to CSV
df.to_csv('sample_data.csv', index=False)
print(f"Sample data created: {df.shape}")
print(df.head())
print(f"\nTarget distribution:\n{df['target'].value_counts()}")
