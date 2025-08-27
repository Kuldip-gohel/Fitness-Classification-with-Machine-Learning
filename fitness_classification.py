import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# 1. load the dataset
fitness = pd.read_csv("fitness_dataset.csv")

# 2. create age catagory
fitness["age_cat"]= pd.cut(
    fitness["age"],
    bins=[0,20,30,40,50,60,70,100],
    labels=[1,2,3,4,5,6,7]
)

# 3. create stratified split for seprate train and test data set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_data, test_data in split.split(fitness, fitness["age_cat"]):
    train_set = fitness.loc[train_data].drop("age_cat", axis=1)
    test_set = fitness.loc[test_data].drop("age_cat", axis=1)

# 4. copy of train_set
fitness = train_set.copy()

# 5.  clean the train set 
# first clean weight_kg column

mean_value = fitness["weight_kg"].mean()

fitness["weight_kg"] = np.where(
    fitness["weight_kg"] > 160,   
    mean_value,                   
    fitness["weight_kg"]        
)

# clean sleep_hours column because it have missing values
fitness["sleep_hours"] = fitness["sleep_hours"].fillna(fitness["sleep_hours"].median())

# clean smoke column because it have mix datatypes values
fitness["smokes"] = fitness["smokes"].replace({
    "yes": 1, "Yes": 1, "YES": 1, "1":1,
    "no": 0, "No": 0, "NO": 0, "0":0
})

# 6. handle catagorical data
fitness["gender"] = fitness["gender"].map({"F": 0, "M": 1})

# 7. seprate feature and label
fitness_labels = fitness["is_fit"].copy()
fitness = fitness.drop("is_fit", axis=1)

# 8. seprate numerical and catagorical column
num_attribs = fitness.drop("gender", axis=1).columns.tolist()
cat_attribs = ["gender"]

# 9. now create pipline
# Numerical pipeline: handle missing, scale values
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Full preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 9.Transform the training data
fitness_prepared = full_pipeline.fit_transform(fitness)

# ----------------------------------------------------------------------------
# # 10. Train Random Forest Classifier
# random_forest_clf = RandomForestClassifier(random_state=42)
# random_forest_clf.fit(fitness_prepared, fitness_labels)

# # Predictions on training set
# rf_preds = random_forest_clf.predict(fitness_prepared)

# # Training accuracy
# train_acc = accuracy_score(fitness_labels, rf_preds)
# print(f"Training Accuracy: {train_acc:.4f}")

# # Cross-validation accuracy
# rf_cv_scores = cross_val_score(random_forest_clf, fitness_prepared, fitness_labels, cv=10, scoring="accuracy")
# print("\nCross-validation accuracy stats:")
# print(pd.Series(rf_cv_scores).describe())

# # Classification report
# print("\nClassification Report:")
# print(classification_report(fitness_labels, rf_preds))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(fitness_labels, rf_preds))

# ------------------------------------------------------------------------------------------------------------
# 10. Train Decision Tree Classifier
# decision_tree_clf = DecisionTreeClassifier(random_state=42)
# decision_tree_clf.fit(fitness_prepared, fitness_labels)

# # Predictions on training set
# dt_preds = decision_tree_clf.predict(fitness_prepared)

# # Training accuracy
# train_acc = accuracy_score(fitness_labels, dt_preds)
# print(f"Training Accuracy: {train_acc:.4f}")

# # Cross-validation accuracy
# dt_cv_scores = cross_val_score(decision_tree_clf, fitness_prepared, fitness_labels, cv=10, scoring="accuracy")
# print("\nCross-validation accuracy stats:")
# print(pd.Series(dt_cv_scores).describe())

# # Classification report
# print("\nClassification Report:")
# print(classification_report(fitness_labels, dt_preds))

# # Confusion matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(fitness_labels, dt_preds))
# ------------------------------------------------------------------------------------------------------------

# 10. Train Logistic Regression model
log_reg_clf = LogisticRegression(random_state=42)
log_reg_clf.fit(fitness_prepared, fitness_labels)

# Predictions on training set
lr_preds = log_reg_clf.predict(fitness_prepared)

# Training accuracy
train_acc = accuracy_score(fitness_labels, lr_preds)
print(f"Training Accuracy: {train_acc:.4f}")

# Cross-validation accuracy
lr_cv_scores = cross_val_score(log_reg_clf, fitness_prepared, fitness_labels, cv=10, scoring="accuracy")
print("\nCross-validation accuracy stats:")
print(pd.Series(lr_cv_scores).describe())

# Classification report
print("\nClassification Report:")
print(classification_report(fitness_labels, lr_preds))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(fitness_labels, lr_preds))






# now work on test data and check model

# 11. Clean the test set 
fitness_test = test_set.copy()
fitness_test["weight_kg"] = np.where(
    fitness_test["weight_kg"] > 160,
    mean_value, 
    fitness_test["weight_kg"]
)

fitness_test["sleep_hours"] = fitness_test["sleep_hours"].fillna(fitness["sleep_hours"].median())

fitness_test["smokes"] = fitness_test["smokes"].replace({
    "yes": 1, "Yes": 1, "YES": 1, "1":1,
    "no": 0, "No": 0, "NO": 0, "0":0
})
fitness_test["gender"] = fitness_test["gender"].map({"F": 0, "M": 1})

# 12. Separate labels and features 
fitness_test_labels = fitness_test["is_fit"].copy()
fitness_test = fitness_test.drop("is_fit", axis=1)

# 13. Apply the same transformations to the test data
fitness_test_prepared = full_pipeline.transform(fitness_test)

# 14. Evaluate on test data
test_preds = log_reg_clf.predict(fitness_test_prepared)

# 15. Check the accuracy and report
test_acc = accuracy_score(fitness_test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(fitness_test_labels, test_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(fitness_test_labels, test_preds))



