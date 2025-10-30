import os
import shutil
import uuid
import pandas as pd
import nbformat
from nbclient import NotebookClient
import copy
import json
import queue
import threading

from fastapi import FastAPI, UploadFile, File, Form, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

DATA_FOLDER = "./dataset"
NOTEBOOKS_FOLDER = "./notebooks"

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(NOTEBOOKS_FOLDER, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


@app.post("/dataset", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)):
    global sessions
    session_id = str(uuid.uuid4())
    file_location = os.path.join(
        DATA_FOLDER, file.filename.split(".")[0] + session_id + ".csv"
    )
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(file_location)
    columns = df.columns.tolist()

    sessions[session_id] = {
        "dataset": file.filename.split(".")[0] + session_id + ".csv",
        "columns": columns,
    }

    return {"columns": columns, "session_id": session_id}


@app.post("/target-column", status_code=status.HTTP_200_OK)
async def set_target_column(session_id: str = Form(...), column_name: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sessions[session_id]["target_column"] = column_name
    df = pd.read_csv(os.path.join(DATA_FOLDER, sessions[session_id]["dataset"]))
    n_unique = df[column_name].nunique()
    if pd.api.types.is_numeric_dtype(df[column_name]) and n_unique > 10:
        return {
            "message": f"{column_name} set successfully as target column.",
            "models": [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "XGBoost Regressor",
                "Support Vector Regressor (SVR)",
                "K-Nearest Neighbors Regressor",
            ],
        }
    return {
        "message": f"{column_name} set successfully as target column.",
        "models": [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Support Vector Machine (SVM)",
            "XGBoost Classifier",
            "K-Nearest Neighbors Classifier",
            "Naive Bayes",
            "Gradient Boosting Classifier",
            "LightGBM Classifier",
            "Extra Trees Classifier",
        ],
    }


@app.post("/model", status_code=status.HTTP_200_OK)
async def set_model(session_id: str = Form(...), model_name: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "target_column" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="Target column not set")
    sessions[session_id]["model"] = model_name
    return {"message": f"{model_name} set successfully as model."}


@app.post("/clean/dataset", status_code=status.HTTP_200_OK)
async def clean_dataset(session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    dataset_path = os.path.join(DATA_FOLDER, sessions[session_id]["dataset"])
    df = pd.read_csv(dataset_path)
    print(df.info())


@app.get("/generate/notebook")
async def generate_notebook(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if (
        "target_column" not in sessions[session_id]
        or "model" not in sessions[session_id]
    ):
        raise HTTPException(status_code=400, detail="Target column or model not set")

    notebook_path = os.path.join(NOTEBOOKS_FOLDER, f"{session_id}.ipynb")
    nb = nbformat.v4.new_notebook()

    # Add title markdown cell
    target_col = sessions[session_id].get("target_column", None)
    model_name = sessions[session_id].get("model", "")
    dataset_name = (
        sessions[session_id]["dataset"].replace(session_id, "").replace(".csv", "")
    )

    title_md = f"""# Machine Learning Pipeline
## Dataset: {dataset_name}
## Target: {target_col}
## Model: {model_name}

This notebook contains a complete end-to-end machine learning pipeline generated automatically."""
    nb.cells.append(nbformat.v4.new_markdown_cell(title_md))

    # Determine if this is a classification or regression problem
    model = model_name.lower()
    is_classifier = (
        ("classifier" in model)
        or ("logistic" in model)
        or ("naive bayes" in model)
        or ("svm" in model and "svr" not in model)
    )

    # Configuration cell
    config_code = f"""# Configuration
# Update these paths if you move the notebook to a different location
SESSION_ID = "{session_id}"
DATA_PATH = r'{os.path.join(DATA_FOLDER, sessions[session_id]["dataset"]).replace("\\", "/")}'
TARGET_COLUMN = "{target_col}"
"""
    nb.cells.append(nbformat.v4.new_code_cell(config_code))

    # Install required packages
    install_md = """## Installation
Run this cell to install all required packages. You can skip this if you already have them installed."""
    nb.cells.append(nbformat.v4.new_markdown_cell(install_md))

    install_code = """%pip install pandas numpy scikit-learn xgboost lightgbm ipykernel -q
print("All packages installed successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(install_code))

    # Import all libraries
    imports_code = """# Import all required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None
print("All libraries imported successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(imports_code))

    # EDA Section
    eda_md = """## 1. Exploratory Data Analysis (EDA)
Let's start by loading and exploring the dataset."""
    nb.cells.append(nbformat.v4.new_markdown_cell(eda_md))

    code1 = """df = pd.read_csv(DATA_PATH)
df.head()"""
    code2 = "# Dataset information\ndf.info()"
    code3 = "# Summary statistics\ndf.describe()"

    # Only show target distribution for classification (not regression)
    if is_classifier:
        code4 = f"""# Check target column distribution
print(f"Target Column: {target_col}")
print(f"\\nClass Distribution:")
counts = df["{target_col}"].value_counts()
for class_name, count in counts.items():
    print(f"  {{class_name}}: {{count}}")
print(f"\\nClass Percentages:")
percentages = df["{target_col}"].value_counts(normalize=True) * 100
for class_name, pct in percentages.items():
    print(f"  {{class_name}}: {{pct:.2f}}%")"""
    else:
        code4 = f"""# Check target column distribution
print(f"Target Column: {target_col}")
print(f"\\nTarget Statistics:")
print(f"  Min: {{df['{target_col}'].min():.2f}}")
print(f"  Max: {{df['{target_col}'].max():.2f}}")
print(f"  Mean: {{df['{target_col}'].mean():.2f}}")
print(f"  Median: {{df['{target_col}'].median():.2f}}")
print(f"  Std Dev: {{df['{target_col}'].std():.2f}}")"""

    code5 = "# Check for missing values\ndf.isnull().sum()"
    # Impute missing values: numerics with median, categoricals with mode or 'Unknown'
    code6 = """impute_report = []
for col in df.columns:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        if np.issubdtype(df[col].dtype, np.number):
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            impute_report.append(f"Filled {n_missing} missing values in '{col}' with median: {median}")
        else:
            mode = df[col].mode().dropna()
            if not mode.empty:
                fill_value = mode[0]
            else:
                fill_value = 'Unknown'
            df[col].fillna(fill_value, inplace=True)
            impute_report.append(f"Filled {n_missing} missing values in '{col}' with mode: {fill_value}")
if impute_report:
    print('Missing value imputation summary:')
    for line in impute_report:
        print(line)
else:
    print('No missing values to impute.')
"""
    nb.cells.append(nbformat.v4.new_code_cell(code1))
    nb.cells.append(nbformat.v4.new_code_cell(code2))
    nb.cells.append(nbformat.v4.new_code_cell(code3))
    nb.cells.append(nbformat.v4.new_code_cell(code4))
    nb.cells.append(nbformat.v4.new_code_cell(code5))

    # Data Cleaning Section
    cleaning_md = """## 2. Data Cleaning
Handle missing values by imputing with appropriate strategies."""
    nb.cells.append(nbformat.v4.new_markdown_cell(cleaning_md))
    nb.cells.append(nbformat.v4.new_code_cell(code6))

    # Feature engineering: one-hot encode categoricals (except target), scale numerics if needed
    model = sessions[session_id].get("model", "").lower()
    tree_models = [
        "decision tree",
        "random forest",
        "xgboost",
        "extra trees",
        "lightgbm",
        "gradient boosting",
    ]
    is_tree = any(tree in model for tree in tree_models)
    is_classifier = (
        ("classifier" in model)
        or ("logistic" in model)
        or ("naive bayes" in model)
        or ("svm" in model)
    )
    needs_scaling = not is_tree
    code6 = f"""feature_report = []
X = df.drop(columns=["{target_col}"], axis=1)
y = df["{target_col}"]

# Identify text columns (high cardinality) and process with TF-IDF
text_cols = []
cat_cols = []
for col in X.select_dtypes(include=["object", "category"]).columns:
    if X[col].nunique() > 50:  # Likely a text column
        text_cols.append(col)
    else:
        cat_cols.append(col)

# Process text columns with TF-IDF
tfidf_features = []
if text_cols:
    for col in text_cols:
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(X[col].fillna('').astype(str))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f"{{col}}_tfidf_{{i}}" for i in range(tfidf_matrix.shape[1])],
            index=X.index
        )
        tfidf_features.append(tfidf_df)
        # Save TF-IDF vectorizer
        with open(f'tfidf_{{col}}_{session_id}.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
    X = X.drop(columns=text_cols)
    feature_report.append(f"Applied TF-IDF to text columns: {{text_cols}} (100 features each)")

# One-hot encode low-cardinality categoricals
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_report.append(f"One-hot encoded columns: {{cat_cols}}")
else:
    feature_report.append("No categorical columns to encode.")

# Combine all features
if tfidf_features:
    X = pd.concat([X] + tfidf_features, axis=1)

# Scale numerics only if needed
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if {str(needs_scaling)}:
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        feature_report.append(f"Scaled numeric columns: {{num_cols}}")
        # Save scaler for future use
        with open('scaler_{session_id}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        feature_report.append("No numeric columns to scale.")
else:
    feature_report.append("Skipped scaling (tree-based model)")

# Save feature names for prediction consistency
with open('features_{session_id}.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

for line in feature_report:
    print(line)
"""
    # Feature Engineering Section
    feature_md = """## 3. Feature Engineering
Transform raw features into model-ready inputs:
- Text columns: TF-IDF vectorization (converts text to numeric features)
- Categorical columns: One-hot encoding
- Numeric columns: Standardization (if needed for the model)"""
    nb.cells.append(nbformat.v4.new_markdown_cell(feature_md))
    nb.cells.append(nbformat.v4.new_code_cell(code6))

    # Train-Test Split Section
    split_md = """## 4. Train-Test Split
Split the data into training (80%) and testing (20%) sets."""
    nb.cells.append(nbformat.v4.new_markdown_cell(split_md))

    # train-test split cell with stratify logic
    if is_classifier:
        code7 = """# Use stratify to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
"""
    else:
        code7 = """X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code7))

    # K-Fold Cross-Validation cell using the selected model
    model_imports = {
        # Classifiers
        "logistic regression": (
            "from sklearn.linear_model import LogisticRegression",
            "LogisticRegression()",
            "accuracy",
        ),
        "decision tree classifier": (
            "from sklearn.tree import DecisionTreeClassifier",
            "DecisionTreeClassifier()",
            "accuracy",
        ),
        "random forest classifier": (
            "from sklearn.ensemble import RandomForestClassifier",
            "RandomForestClassifier()",
            "accuracy",
        ),
        "support vector machine (svm)": (
            "from sklearn.svm import SVC",
            "SVC()",
            "accuracy",
        ),
        "xgboost classifier": (
            "from xgboost import XGBClassifier",
            "XGBClassifier(eval_metric='logloss', verbosity=0)",
            "accuracy",
        ),
        "k-nearest neighbors classifier": (
            "from sklearn.neighbors import KNeighborsClassifier",
            "KNeighborsClassifier()",
            "accuracy",
        ),
        "naive bayes": (
            "from sklearn.naive_bayes import GaussianNB",
            "GaussianNB()",
            "accuracy",
        ),
        "gradient boosting classifier": (
            "from sklearn.ensemble import GradientBoostingClassifier",
            "GradientBoostingClassifier()",
            "accuracy",
        ),
        "lightgbm classifier": (
            "from lightgbm import LGBMClassifier",
            "LGBMClassifier(verbosity=-1)",
            "accuracy",
        ),
        "extra trees classifier": (
            "from sklearn.ensemble import ExtraTreesClassifier",
            "ExtraTreesClassifier()",
            "accuracy",
        ),
        # Regressors
        "linear regression": (
            "from sklearn.linear_model import LinearRegression",
            "LinearRegression()",
            "r2",
        ),
        "ridge regression": ("from sklearn.linear_model import Ridge", "Ridge()", "r2"),
        "lasso regression": ("from sklearn.linear_model import Lasso", "Lasso()", "r2"),
        "decision tree regressor": (
            "from sklearn.tree import DecisionTreeRegressor",
            "DecisionTreeRegressor()",
            "r2",
        ),
        "random forest regressor": (
            "from sklearn.ensemble import RandomForestRegressor",
            "RandomForestRegressor()",
            "r2",
        ),
        "gradient boosting regressor": (
            "from sklearn.ensemble import GradientBoostingRegressor",
            "GradientBoostingRegressor()",
            "r2",
        ),
        "xgboost regressor": (
            "from xgboost import XGBRegressor",
            "XGBRegressor(verbosity=0)",
            "r2",
        ),
        "support vector regressor (svr)": (
            "from sklearn.svm import SVR",
            "SVR()",
            "r2",
        ),
        "k-nearest neighbors regressor": (
            "from sklearn.neighbors import KNeighborsRegressor",
            "KNeighborsRegressor()",
            "r2",
        ),
    }

    model_key = sessions[session_id].get("model", "").strip().lower()
    import_stmt, model_ctor, scoring = model_imports.get(
        model_key,
        (
            "from sklearn.linear_model import LogisticRegression",
            "LogisticRegression()",
            "accuracy" if is_classifier else "r2",
        ),
    )

    # Cross-Validation Section
    cv_md = """## 5. Cross-Validation
Evaluate model performance using 5-fold cross-validation on the training set.
This helps assess how well the model generalizes."""
    nb.cells.append(nbformat.v4.new_markdown_cell(cv_md))

    if is_classifier:
        code8 = f"""# Use stratified K-fold to maintain class distribution in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv = {model_ctor}
scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring="{scoring}")
print(f"K-Fold CV {scoring.upper()}: {{np.mean(scores) * 100:.2f}}% ± {{np.std(scores) * 100:.2f}}%")
"""
    else:
        code8 = f"""# Use K-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
model_cv = {model_ctor}
scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring="{scoring}")
print(f"K-Fold CV {scoring.title()}: {{np.mean(scores):.4f}} ± {{np.std(scores):.4f}}")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code8))

    # Model Training and Evaluation Section
    eval_md = """## 6. Final Model Training and Evaluation
Train the final model on the training set and evaluate on the test set."""
    nb.cells.append(nbformat.v4.new_markdown_cell(eval_md))

    # Model fitting and test set evaluation cell
    if is_classifier:
        code9 = f"""# Train the model
model = {model_ctor}
model.fit(X_train, y_train)

# Evaluate on training set
X_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, X_train_pred)

# Evaluate on test set
X_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, X_test_pred)

print(f"Train Accuracy: {{train_acc * 100:.2f}}%")
print(f"Test Accuracy: {{test_acc * 100:.2f}}%")
print(f"\\nClassification Report:\\n{{classification_report(y_test, X_test_pred)}}")

# Save trained model
with open('model_{session_id}.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\\nModel saved successfully!")
"""
    else:
        code9 = f"""# Train the model
model = {model_ctor}
model.fit(X_train, y_train)

# Evaluate on training set
X_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, X_train_pred)

# Evaluate on test set
X_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, X_test_pred)
test_mse = mean_squared_error(y_test, X_test_pred)

print(f"Train R2: {{train_r2:.4f}}")
print(f"Test R2: {{test_r2:.4f}}")
print(f"Test MSE: {{test_mse:.4f}}")

# Save trained model
with open('model_{session_id}.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\\nModel saved successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code9))

    # Summary and Usage Instructions
    summary_md = f"""## 7. Summary and Next Steps

### Model Artifacts Saved:
- `model_{session_id}.pkl` - Trained model
- `scaler_{session_id}.pkl` - Feature scaler
- TF-IDF vectorizers (if text columns were present)

### How to Use the Saved Model:

```python
import pickle
import pandas as pd

# Load the trained model
with open('model_{session_id}.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler_{session_id}.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load new data
new_data = pd.read_csv('your_new_data.csv')

# Preprocess (apply the same transformations)
# ... apply same one-hot encoding, TF-IDF, and scaling ...

# Make predictions
predictions = model.predict(new_data_processed)
```

### Model Performance:
- Cross-validation helps ensure the model generalizes well
- Compare train vs test performance to check for overfitting
- If test performance is significantly lower, consider:
  - Collecting more data
  - Feature engineering
  - Regularization
  - Different model architecture

**Note:** To use this notebook on a different machine, update the `DATA_PATH` in the configuration cell at the top."""
    nb.cells.append(nbformat.v4.new_markdown_cell(summary_md))

    # Track model artifacts in session
    sessions[session_id]["artifacts"] = {
        "model": f"model_{session_id}.pkl",
        "scaler": f"scaler_{session_id}.pkl",
        "notebook": f"{session_id}.ipynb",
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    def sse_event_stream():
        result_queue = queue.Queue()
        code_cell_counter = [0]  # Use list to make it mutable in closure

        def on_cell_executed(cell=None, cell_index=None, **kwargs):
            """Hook called after each cell execution"""
            # Only process code cells, skip markdown cells
            if not cell or cell.cell_type != "code":
                return

            code_cell_counter[0] += 1

            # Extract result from cell outputs - concatenate all outputs
            cell_result = None
            if cell.outputs:
                results = []
                for output in cell.outputs:
                    if output.get("output_type") == "execute_result":
                        result = output["data"].get("text/html") or output["data"].get(
                            "text/plain"
                        )
                        if result:
                            results.append(result)
                    elif output.get("output_type") == "stream":
                        result = output.get("text")
                        if result:
                            results.append(result)
                    elif output.get("output_type") == "display_data":
                        result = output["data"].get("text/html") or output["data"].get(
                            "text/plain"
                        )
                        if result:
                            results.append(result)
                    elif output.get("output_type") == "error":
                        # Capture error output
                        error_name = output.get("ename", "Error")
                        error_value = output.get("evalue", "")
                        results.append(f"{error_name}: {error_value}")

                # Concatenate all outputs
                if results:
                    cell_result = (
                        "".join(results)
                        if all(isinstance(r, str) for r in results)
                        else results[0]
                    )
            else:
                # Debug: log when cell has no outputs
                print(f"DEBUG: Cell {code_cell_counter[0]} has no outputs")

            # Put result in queue for streaming
            result_queue.put({"cell": code_cell_counter[0], "result": cell_result})

        def execute_notebook():
            """Execute notebook in a separate thread"""
            try:
                nb_to_run = copy.deepcopy(nb)
                client = NotebookClient(nb_to_run, allow_errors=True, timeout=600)

                # Register hook to stream results after each cell
                client.on_cell_executed = on_cell_executed

                # Execute the entire notebook (hook will stream results)
                client.execute()

                result_queue.put(None)  # Signal completion

            except Exception as e:
                import traceback

                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                result_queue.put({"error": error_msg})
                result_queue.put(None)  # Signal completion

        # Start execution in background thread
        thread = threading.Thread(target=execute_notebook, daemon=True)
        thread.start()

        # Stream results as they become available
        while True:
            try:
                result = result_queue.get(timeout=2)
                if result is None:
                    break
                yield "data: {}\n\n".format(json.dumps(result))
            except queue.Empty:
                # Send keepalive to prevent timeout
                yield ": keepalive\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.get("/download/model/{session_id}")
async def download_model(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "artifacts" not in sessions[session_id]:
        raise HTTPException(
            status_code=404, detail="No model artifacts found for this session"
        )

    model_file = sessions[session_id]["artifacts"]["model"]
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_file,
        media_type="application/octet-stream",
        filename=f"model_{session_id}.pkl",
    )


@app.get("/download/scaler/{session_id}")
async def download_scaler(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "artifacts" not in sessions[session_id]:
        raise HTTPException(
            status_code=404, detail="No scaler artifacts found for this session"
        )

    scaler_file = sessions[session_id]["artifacts"]["scaler"]
    if not os.path.exists(scaler_file):
        raise HTTPException(status_code=404, detail="Scaler file not found")

    return FileResponse(
        scaler_file,
        media_type="application/octet-stream",
        filename=f"scaler_{session_id}.pkl",
    )


@app.get("/download/notebook/{session_id}")
async def download_notebook(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "artifacts" not in sessions[session_id]:
        raise HTTPException(
            status_code=404, detail="No notebook artifacts found for this session"
        )

    notebook_file = os.path.join(
        NOTEBOOKS_FOLDER, sessions[session_id]["artifacts"]["notebook"]
    )
    if not os.path.exists(notebook_file):
        raise HTTPException(status_code=404, detail="Notebook file not found")

    return FileResponse(
        notebook_file,
        media_type="application/x-ipynb+json",
        filename=f"ml_notebook_{session_id}.ipynb",
    )


@app.post("/predict/{session_id}")
async def predict(session_id: str, input_data: dict):
    import pickle
    import numpy as np

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "artifacts" not in sessions[session_id]:
        raise HTTPException(status_code=404, detail="Model not trained yet")

    model_file = sessions[session_id]["artifacts"]["model"]
    scaler_file = sessions[session_id]["artifacts"]["scaler"]

    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        # Load model
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Load scaler if exists
        scaler = None
        if os.path.exists(scaler_file):
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)

        # Get original dataset to understand feature types
        df = pd.read_csv(os.path.join(DATA_FOLDER, sessions[session_id]["dataset"]))
        target_col = sessions[session_id]["target_column"]
        X_original = df.drop(columns=[target_col])

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_data])

        # Fill missing columns with median/mode from training data
        for col in X_original.columns:
            if col not in input_df.columns:
                if pd.api.types.is_numeric_dtype(X_original[col]):
                    input_df[col] = X_original[col].median()
                else:
                    mode_val = X_original[col].mode()
                    input_df[col] = mode_val[0] if not mode_val.empty else "Unknown"

        # Keep only columns that exist in original dataset
        input_df = input_df[
            [col for col in X_original.columns if col in input_df.columns]
        ]

        # Apply same preprocessing as training
        # 1. Identify text and categorical columns
        text_cols = []
        cat_cols = []
        for col in input_df.select_dtypes(include=["object"]).columns:
            if X_original[col].nunique() > 50:
                text_cols.append(col)
            else:
                cat_cols.append(col)

        # 2. Process text columns with TF-IDF
        tfidf_features = []
        if text_cols:
            for col in text_cols:
                tfidf_file = f"tfidf_{col}_{session_id}.pkl"
                if os.path.exists(tfidf_file):
                    with open(tfidf_file, "rb") as f:
                        tfidf = pickle.load(f)
                    tfidf_matrix = tfidf.transform(input_df[col].fillna("").astype(str))
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[
                            f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])
                        ],
                        index=input_df.index,
                    )
                    tfidf_features.append(tfidf_df)
            input_df = input_df.drop(columns=text_cols)

        # 3. One-hot encode categorical columns
        if cat_cols:
            input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # 4. Combine all features
        if tfidf_features:
            input_df = pd.concat([input_df] + tfidf_features, axis=1)

        # 5. Align columns with training data
        # Load saved feature names from training
        features_file = f"features_{session_id}.pkl"
        if os.path.exists(features_file):
            with open(features_file, "rb") as f:
                expected_features = pickle.load(f)

            # Add missing columns with 0
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Remove extra columns and reorder to match training
            input_df = input_df[expected_features]
        else:
            # Fallback: try to get from model
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
                for col in expected_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[expected_features]

        # 6. Scale numeric features if scaler exists
        if scaler:
            num_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Convert to array
        X = input_df.values

        # Predict
        prediction = model.predict(X)[0]

        # Get probability for classifiers if available
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            probability = {str(i): float(p) for i, p in enumerate(proba)}

        return {
            "prediction": float(prediction),
            "probability": probability,
            "input_processed": input_df.columns.tolist(),
        }
    except Exception as e:
        import traceback

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}\n{traceback.format_exc()}",
        )


@app.delete("/cleanup/all")
async def cleanup_all():
    """Clean up ALL sessions and files"""
    try:
        global sessions
        files_deleted = []

        # Delete all files in data folder
        for file in os.listdir(DATA_FOLDER):
            file_path = os.path.join(DATA_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_deleted.append(file_path)

        # Delete all files in notebooks folder
        for file in os.listdir(NOTEBOOKS_FOLDER):
            file_path = os.path.join(NOTEBOOKS_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_deleted.append(file_path)

        # Delete all pkl files in root (models, scalers, tfidf)
        for file in os.listdir("."):
            if file.endswith(".pkl"):
                os.remove(file)
                files_deleted.append(file)

        # Clear sessions dictionary
        session_count = len(sessions)
        sessions = {}

        return {
            "message": "All sessions cleaned up successfully",
            "sessions_cleared": session_count,
            "files_deleted": files_deleted,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with their details"""
    session_list = []
    for sid, data in sessions.items():
        session_info = {
            "session_id": sid,
            "dataset": data.get("dataset", "N/A"),
            "target_column": data.get("target_column", "N/A"),
            "model": data.get("model", "N/A"),
            "has_artifacts": "artifacts" in data,
        }
        session_list.append(session_info)

    return {"total_sessions": len(sessions), "sessions": session_list}


@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up all files and data for a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        session_data = sessions[session_id]
        files_deleted = []

        # Delete dataset file
        if "dataset" in session_data:
            dataset_path = os.path.join(DATA_FOLDER, session_data["dataset"])
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                files_deleted.append(dataset_path)

        # Delete artifacts (model, scaler, tfidf, notebook)
        if "artifacts" in session_data:
            artifacts = session_data["artifacts"]

            # Delete model
            if "model" in artifacts and os.path.exists(artifacts["model"]):
                os.remove(artifacts["model"])
                files_deleted.append(artifacts["model"])

            # Delete scaler
            if "scaler" in artifacts and os.path.exists(artifacts["scaler"]):
                os.remove(artifacts["scaler"])
                files_deleted.append(artifacts["scaler"])

            # Delete notebook
            if "notebook" in artifacts:
                notebook_path = os.path.join(NOTEBOOKS_FOLDER, artifacts["notebook"])
                if os.path.exists(notebook_path):
                    os.remove(notebook_path)
                    files_deleted.append(notebook_path)

        # Delete any TF-IDF vectorizer and feature files
        for file in os.listdir("."):
            if (
                file.startswith("tfidf_") or file.startswith("features_")
            ) and file.endswith(f"_{session_id}.pkl"):
                os.remove(file)
                files_deleted.append(file)

        # Remove session from memory
        del sessions[session_id]

        return {
            "message": "Session cleaned up successfully",
            "files_deleted": files_deleted,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
