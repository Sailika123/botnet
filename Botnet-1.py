import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns   # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import svm # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.ensemble import GradientBoostingClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import pickle  # Import pickle for saving models

# Load dataset
dataset = pd.read_csv(r"C:\Botnet\UNSW_2018_IoT_Botnet_Final_10_best_Testing - Copy.csv")

# Display initial dataset information
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset.isna().sum())

# Define categorical columns
categorical_columns = ['proto', 'saddr', 'daddr', 'category', 'subcategory']
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ['attack']]    # Exclude target variable

# Preprocessing
# Create transformers
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Scale numerical features
])

# Apply transformers based on feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Separate features and target variable
X = dataset.drop('attack', axis=1)
y = dataset['attack']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Preprocess the data
x_train_preprocessed = preprocessor.fit_transform(x_train)

# SVM Classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(x_train_preprocessed, y_train)

# Predictions and Accuracy for SVM
svm_train_prediction = svm_classifier.predict(x_train_preprocessed)
svm_training_data_accuracy = accuracy_score(y_train, svm_train_prediction)
print('Accuracy score for SVM train data:', svm_training_data_accuracy * 100)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(x_train_preprocessed, y_train)

# Predictions and Accuracy for Random Forest
rf_train_prediction = rf_classifier.predict(x_train_preprocessed)
rf_training_data_accuracy = accuracy_score(y_train, rf_train_prediction)
print('Accuracy score for RFC train data:', rf_training_data_accuracy * 100)

# Logistic Regression Classifier
lr_classifier = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
lr_classifier.fit(x_train_preprocessed, y_train)

# Predictions and Accuracy for Logistic Regression
lr_train_prediction = lr_classifier.predict(x_train_preprocessed)
lr_training_data_accuracy = accuracy_score(y_train, lr_train_prediction)
print('Accuracy score for Logistic Regression train data:', lr_training_data_accuracy * 100)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(x_train_preprocessed, y_train)

# Predictions and Accuracy for K-Nearest Neighbors
knn_train_prediction = knn_classifier.predict(x_train_preprocessed)
knn_training_data_accuracy = accuracy_score(y_train, knn_train_prediction)
print('Accuracy score for KNN train data:', knn_training_data_accuracy * 100)

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(x_train_preprocessed, y_train)

# Predictions and Accuracy for Gradient Boosting
gb_train_prediction = gb_classifier.predict(x_train_preprocessed)
gb_training_data_accuracy = accuracy_score(y_train, gb_train_prediction)
print('Accuracy score for Gradient Boosting train data:', gb_training_data_accuracy * 100)


from sklearn.model_selection import GridSearchCV # type: ignore

# Define hyperparameter grids for each classifier
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

lr_param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'saga', 'liblinear']  # Make sure solvers match penalties
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}



# Instantiate GridSearchCV for each model
knn_grid_search = GridSearchCV(knn_classifier, knn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV to the training data
knn_grid_search.fit(x_train_preprocessed, y_train)

# Get the best parameters and accuracy for each model



print("Best KNN parameters:", knn_grid_search.best_params_)
print("Best KNN accuracy:", knn_grid_search.best_score_)


best_knn = knn_grid_search.best_estimator_

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== GridSearchCV for SVM ==========
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svm_grid_search = GridSearchCV(svm.SVC(), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(x_train_preprocessed, y_train)
print("Best SVM parameters:", svm_grid_search.best_params_)
print("Best SVM accuracy:", svm_grid_search.best_score_)

# ========== GridSearchCV for Random Forest ==========
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(x_train_preprocessed, y_train)
print("Best Random Forest parameters:", rf_grid_search.best_params_)
print("Best RF accuracy:", rf_grid_search.best_score_)

# ========== GridSearchCV for Logistic Regression ==========
lr_param_grid = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000), lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid_search.fit(x_train_preprocessed, y_train)
print("Best Logistic Regression parameters:", lr_grid_search.best_params_)
print("Best LR accuracy:", lr_grid_search.best_score_)

# ========== Save All Best Models ==========
with open('best_svm_model.pkl', 'wb') as f:
    pickle.dump(svm_grid_search.best_estimator_, f)

with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_grid_search.best_estimator_, f)

with open('best_lr_model.pkl', 'wb') as f:
    pickle.dump(lr_grid_search.best_estimator_, f)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# ========== Visualization 1: Correlation Heatmap ==========
# Filter only numeric columns before correlation
numeric_data = dataset.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ========== Visualization 2: Class Distribution ==========
sns.countplot(x='attack', data=dataset)
plt.title("Distribution of Attack vs Normal")
plt.xlabel("Attack (1) / Normal (0)")
plt.ylabel("Count")
plt.show()

# ========== Visualization 3: Confusion Matrix for Best KNN ==========
x_test_preprocessed = preprocessor.transform(x_test)
knn_test_prediction = best_knn.predict(x_test_preprocessed)

cm = confusion_matrix(y_test, knn_test_prediction)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Best KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




