"""
Machine Learning Data Preprocessing and Impact on Model Performance

This script demonstrates the importance of feature scaling in machine learning.
We generate a synthetic dataset with features on different scales and show how
scaling impacts the accuracy of a K-Nearest Neighbors (KNN) classifier.

Key Steps:
1. Generate synthetic data with varying scales.
2. Save and reload the dataset from a CSV file.
3. Split the data into training and test sets.
4. Train a KNN classifier without scaling and measure accuracy.
5. Apply feature scaling using StandardScaler.
6. Train KNN again with scaled data and compare accuracy.

Feature scaling is crucial for distance-based algorithms like KNN to ensure
that all features contribute equally to the model.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset from CSV
dataset = pd.read_csv("audible_uncleaned.csv")  # Ensure dataset.csv is present in the working directory

# Display first 5 rows of raw data
print("Raw Data (First 5 rows):")
print(dataset.head().to_string(index=False))
print("\n" + "="*50 + "\n")

# Split features and labels
X = dataset[['Feature1', 'Feature2']].values
y = dataset['Label'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train without scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_no_scaling = accuracy_score(y_test, y_pred)

# Preprocess data with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled data samples
print("Scaled Data Example (First 5 training samples):")
print(pd.DataFrame(X_train_scaled, columns=['Scaled_F1', 'Scaled_F2']).head().to_string(index=False))
print("\n" + "="*50 + "\n")

# Train with scaling
knn.fit(X_train_scaled, y_train)
y_pred_scaled = knn.predict(X_test_scaled)
accuracy_with_scaling = accuracy_score(y_test, y_pred_scaled)

# Results comparison
print(f"Accuracy without scaling: {accuracy_no_scaling:.2f}")
print(f"Accuracy with scaling: {accuracy_with_scaling:.2f}")
print("Improvement: {:.0f}%".format((accuracy_with_scaling - accuracy_no_scaling)*100))


"""
Observations:
- Before scaling, the model is likely to be biased towards the larger-scale feature.
- After scaling, features contribute equally, leading to better model performance.
- Feature scaling is essential for distance-based algorithms like KNN.
"""
