
import os
import zipfile
import urllib.request
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

os.makedirs("data", exist_ok=True)

zip_path = "data/bank-marketing.zip"
urllib.request.urlretrieve(url, zip_path)

def find_csv(zip_data):
    from io import BytesIO

    with zipfile.ZipFile(zip_data) as z:
        for name in z.namelist():
            if os.path.basename(name).lower() == "bank-full.csv":
                return z.read(name)

        for name in z.namelist():
            if name.lower().endswith(".zip"):
                result = find_csv(BytesIO(z.read(name)))
                if result is not None:
                    return result

    return None

from io import BytesIO

with open(zip_path, "rb") as f:
    csv_data = find_csv(BytesIO(f.read()))

if csv_data is None:
    raise FileNotFoundError("bank-full.csv not found")

df = pd.read_csv(BytesIO(csv_data), sep=";")

df = df.drop(columns=["duration"])
df["y"] = df["y"].map({"yes": 1, "no": 0})

X = df.drop(columns=["y"])
y = df["y"]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

models = [
    (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"model__C": [0.1, 1, 10]}
    ),
    (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"model__max_depth": [5, 10, 20]}
    ),
    (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [10, 20]}
    )
]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_score = 0
best_model = None

for model, params in models:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        params,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1
    )

    grid.fit(X, y)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_

joblib.dump(best_model, "model.bin")

print("Training completed.")
print("Best CV ROC-AUC:", best_score)
