# load pre-trained data
# Imputing & Encoding
# Train-test split
# Model Training & Hyperparameter Tuning
# Model Evaluation and select the best model
# dump best model

#Imports

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load pre-cleaned data 
CLEANED_DATA_PATH = r"C:\Users\hp\Downloads\BIA\Customer_Churn _Prediction\Data\cleaned_data.csv"
data = pd.read_csv(CLEANED_DATA_PATH)
data.head()
print("Data loaded successfully.")

# Data back-up
df = data.copy()

# Feature and Target
X = df.drop(columns = ["Churn", "Churn_encoded"], axis=1)
y = df["Churn_encoded"]

num_cols = X.select_dtypes(include = "number").columns.to_list()
cat_cols = X.select_dtypes(include = "object").columns.to_list()

# Imputing & Encoding
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])   

cat_pipeline = Pipeline(steps= [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first" ))
])
preprocessor = ColumnTransformer(transformers= 
    [("num_transformer", num_pipeline, num_cols),
     ("cat_transformer", cat_pipeline, cat_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-test split completed.")

# Model Dictionary
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1]}
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20]
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(),
        {
            "model__n_estimators": [50, 100, 200]
        }
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss"),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5, 7]
        }
    ),
    "SVC": (
        SVC(),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear", "poly"]
        }
    )
}

result = []
best_model = None
best_score = 0

# Training
for name, (model,param) in models.items():
    print(f"model: {name} is running...")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc= metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    result.append({
    "Model": name,
    "Accuracyy" : acc,
    "F1_Score":f1
 })

    if f1 > best_score:
       best_score = f1
       best_model = grid.best_estimator_

result_df = pd.DataFrame(result) 

# Sort dataframe about f1_score
result_df = result_df.sort_values(by="F1_Score", ascending=False)

print("Model Comparison:")
print(result_df)

print("\nBest Model:", result_df.iloc[0]["Model"])

MODEL_PATH= r"C:\Users\hp\Downloads\BIA\Customer_Churn _Prediction\best_model.pkl"

# Dump the model
joblib.dump(best_model, MODEL_PATH)
print("Model Trained and Saved Sucessfully.")
