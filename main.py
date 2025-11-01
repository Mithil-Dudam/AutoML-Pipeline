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
import asyncio

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    status,
    HTTPException,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from langchain_ollama import ChatOllama

DATA_FOLDER = "./dataset"
NOTEBOOKS_FOLDER = "./notebooks"
ARTIFACTS_FOLDER = "./artifacts"
llm = ChatOllama(model="llama3.2", temperature=0)

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(NOTEBOOKS_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

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

    # Validate file is a CSV
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Validate file size (e.g., max 100MB)
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(file_content) > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(
            status_code=400, detail="File too large. Maximum size is 100MB"
        )

    session_id = str(uuid.uuid4())
    filename_base = (
        file.filename.rsplit(".", 1)[0] if "." in file.filename else file.filename
    )
    file_location = os.path.join(DATA_FOLDER, filename_base + session_id + ".csv")

    # Save file
    with open(file_location, "wb") as buffer:
        buffer.write(file_content)

    # Try to read and validate CSV
    try:
        df = pd.read_csv(file_location)
    except pd.errors.EmptyDataError:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    # Validate dataframe is not empty
    if df.empty:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="CSV file contains no data")

    if len(df.columns) == 0:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="CSV file contains no columns")

    # Check for minimum rows
    if len(df) < 10:
        os.remove(file_location)
        raise HTTPException(
            status_code=400,
            detail="CSV file must contain at least 10 rows for meaningful analysis",
        )

    columns = df.columns.tolist()

    sessions[session_id] = {
        "dataset": filename_base + session_id + ".csv",
        "columns": columns,
    }

    return {"columns": columns, "session_id": session_id}


@app.post("/target-column", status_code=status.HTTP_200_OK)
async def set_target_column(session_id: str = Form(...), column_name: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Read dataset
    dataset_path = os.path.join(DATA_FOLDER, sessions[session_id]["dataset"])
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    # Validate column exists
    if column_name not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column_name}' not found in dataset. Available columns: {', '.join(df.columns.tolist())}",
        )

    # Validate column has data
    if df[column_name].isnull().all():
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column_name}' contains only missing values",
        )

    # Validate sufficient non-null values
    non_null_count = df[column_name].notna().sum()
    if non_null_count < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column_name}' has only {non_null_count} non-null values. Need at least 5 for training",
        )

    sessions[session_id]["target_column"] = column_name
    n_unique = df[column_name].nunique()

    # Analyze target distribution for warnings and recommendations
    warnings = []
    recommendations = []

    # Phase 3: Smart Column Detection - Detect columns that should be dropped
    problematic_columns = {
        "id_columns": [],
        "email_columns": [],
        "name_columns": [],
        "url_columns": [],
        "phone_columns": [],
        "hash_columns": [],
    }

    for col in df.columns:
        if col == column_name:  # Skip target column
            continue

        col_lower = col.lower()
        col_data = df[col].astype(str)
        col_n_unique = df[
            col
        ].nunique()  # Renamed to avoid overwriting target's n_unique
        uniqueness_ratio = col_n_unique / len(df)

        # Track if column has been classified (to avoid checking multiple categories)
        classified = False

        # 1. Email Columns (check name first)
        if any(keyword in col_lower for keyword in ["email", "e-mail", "mail"]):
            problematic_columns["email_columns"].append(col)
            classified = True
        # Check if data looks like emails
        elif not classified and df[col].dtype == "object":
            sample = col_data.dropna().head(10)
            if len(sample) > 0 and sample.str.contains("@").sum() > len(sample) * 0.7:
                problematic_columns["email_columns"].append(col)
                classified = True

        # 2. URL/File Path Columns
        if not classified and any(
            keyword in col_lower for keyword in ["url", "link", "path", "uri", "href"]
        ):
            problematic_columns["url_columns"].append(col)
            classified = True
        # Check if data looks like URLs
        elif not classified and df[col].dtype == "object":
            sample = col_data.dropna().head(10)
            if len(sample) > 0:
                url_pattern = sample.str.contains(
                    "http://|https://|www\\.|\\.com|\\.org", regex=True
                )
                if url_pattern.sum() > len(sample) * 0.5:
                    problematic_columns["url_columns"].append(col)
                    classified = True

        # 3. Phone Number Columns
        if not classified and any(
            keyword in col_lower
            for keyword in ["phone", "mobile", "tel", "fax", "contact"]
        ):
            problematic_columns["phone_columns"].append(col)
            classified = True

        # 4. Name Columns
        if not classified and any(
            keyword in col_lower
            for keyword in ["name", "firstname", "lastname", "fullname", "username"]
        ):
            if uniqueness_ratio > 0.5:  # High cardinality names
                problematic_columns["name_columns"].append(col)
                classified = True

        # 5. Hash/Token Columns (check before ID since they're more specific)
        if not classified and df[col].dtype == "object" and uniqueness_ratio > 0.98:
            sample = (
                col_data.dropna().head(5).iloc[0] if len(col_data.dropna()) > 0 else ""
            )
            if len(str(sample)) > 20:  # Long strings
                problematic_columns["hash_columns"].append(col)
                classified = True

        # 6. ID Columns (check last as it's the most generic)
        # Only flag as ID if column name contains 'id' (with word boundary or underscore) OR it's a string column with >95% unique values
        # Don't flag numeric columns with high uniqueness (they could be useful features)
        # Use word boundaries to avoid false positives like "acidity", "citric acid", "chlorides"
        import re

        # Match 'id' at word boundaries or with underscores: customer_id, user_id, id, ID, etc.
        if not classified and re.search(r"(?:^|_)id(?:$|_)", col_lower):
            problematic_columns["id_columns"].append(col)
            classified = True
        elif not classified and df[col].dtype == "object" and uniqueness_ratio > 0.95:
            problematic_columns["id_columns"].append(col)
            classified = True

    # Combine all problematic columns
    all_problematic = []
    for category, cols in problematic_columns.items():
        all_problematic.extend(cols)

    # Create warnings for detected columns
    if all_problematic:
        warning_messages = []
        if problematic_columns["id_columns"]:
            warning_messages.append(
                f"🆔 ID columns: {', '.join(problematic_columns['id_columns'])}"
            )
        if problematic_columns["email_columns"]:
            warning_messages.append(
                f"📧 Email columns: {', '.join(problematic_columns['email_columns'])}"
            )
        if problematic_columns["name_columns"]:
            warning_messages.append(
                f"👤 Name columns: {', '.join(problematic_columns['name_columns'])}"
            )
        if problematic_columns["url_columns"]:
            warning_messages.append(
                f"🔗 URL columns: {', '.join(problematic_columns['url_columns'])}"
            )
        if problematic_columns["phone_columns"]:
            warning_messages.append(
                f"📱 Phone columns: {', '.join(problematic_columns['phone_columns'])}"
            )
        if problematic_columns["hash_columns"]:
            warning_messages.append(
                f"🔐 Hash/Token columns: {', '.join(problematic_columns['hash_columns'])}"
            )

        warnings.append(
            {
                "type": "problematic_columns_detected",
                "severity": "medium",
                "message": f"📌 {len(all_problematic)} column(s) detected with low predictive value (will be auto-dropped)",
                "details": warning_messages,
                "columns": all_problematic,
                "impact": "These columns typically have no predictive value and add noise to the model. They will be automatically excluded during training.",
            }
        )
        # Store all problematic columns in session
        sessions[session_id]["id_columns"] = all_problematic
        sessions[session_id]["problematic_column_details"] = problematic_columns

    # Check if this is regression or classification
    is_regression = pd.api.types.is_numeric_dtype(df[column_name]) and n_unique > 10

    if is_regression:
        # Regression - return with ID warnings if any
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
            "warnings": warnings,
            "recommendations": recommendations,
        }

    # Classification - analyze class distribution
    value_counts = df[column_name].value_counts().sort_index()
    total_samples = len(df)
    n_classes = len(value_counts)

    # Validate classification data
    if n_classes < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{column_name}' must have at least 2 unique classes for classification. Found only {n_classes}",
        )

    if n_classes > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{column_name}' has {n_classes} unique classes. This is too many for classification. Consider regression or reducing classes.",
        )

    # Calculate class imbalance metrics
    max_class_count = value_counts.max()
    min_class_count = value_counts.min()
    imbalance_ratio = (
        max_class_count / min_class_count if min_class_count > 0 else float("inf")
    )

    # Find minority classes (< 5% of data)
    minority_threshold = total_samples * 0.05
    minority_classes = value_counts[value_counts < minority_threshold]

    # Detect small dataset with many classes
    samples_per_class = total_samples / n_classes

    # Generate warnings
    if len(minority_classes) > 0:
        minority_details = {str(k): int(v) for k, v in minority_classes.items()}
        warnings.append(
            {
                "type": "class_imbalance",
                "severity": "high"
                if len(minority_classes) > n_classes / 2
                else "medium",
                "message": f"⚠️ {len(minority_classes)} out of {n_classes} classes have fewer than 5% of samples",
                "details": minority_details,
                "impact": "Model may ignore minority classes and predict only majority classes",
            }
        )

    if samples_per_class < 50 and n_classes >= 5:
        warnings.append(
            {
                "type": "small_dataset",
                "severity": "high",
                "message": f"⚠️ Only ~{int(samples_per_class)} samples per class (need 50+ for reliable training)",
                "details": {
                    "total_samples": total_samples,
                    "n_classes": n_classes,
                    "samples_per_class": int(samples_per_class),
                },
                "impact": "High risk of overfitting - model memorizes training data but fails on new data",
            }
        )

    if imbalance_ratio > 10:
        warnings.append(
            {
                "type": "severe_imbalance",
                "severity": "high",
                "message": f"⚠️ Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)",
                "details": {
                    "max_class_samples": int(max_class_count),
                    "min_class_samples": int(min_class_count),
                    "ratio": round(imbalance_ratio, 2),
                },
                "impact": "Model will be biased toward majority class",
            }
        )

    # Generate recommendations for binary transformation
    if n_classes >= 3 and pd.api.types.is_numeric_dtype(df[column_name]):
        # Suggest binary transformation if conditions are met
        should_suggest = (
            (
                len(minority_classes) > 0
                and n_classes
                >= 3  # Changed from >= 5 to >= 3 for wine quality (3-8 scale)
            )  # Many classes with imbalance
            or (
                samples_per_class < 50 and n_classes >= 5
            )  # Small dataset with many classes
            or (imbalance_ratio > 10)  # Severe imbalance
            or (
                n_classes >= 5
            )  # 5+ classes almost always benefit from binary simplification
        )

        if should_suggest:
            # Determine best threshold (75th percentile for "good" class)
            threshold = df[column_name].quantile(0.75)

            # Calculate new distribution after transformation
            positive_count = (df[column_name] >= threshold).sum()
            negative_count = (df[column_name] < threshold).sum()
            new_balance = (
                min(positive_count, negative_count)
                / max(positive_count, negative_count)
                * 100
            )

            recommendations.append(
                {
                    "type": "binary_transformation",
                    "priority": "high",
                    "title": "💡 Consider Simplifying to Binary Classification",
                    "message": f"Transform {column_name} into 2 classes for better model performance",
                    "suggestion": f"{column_name} >= {threshold:.1f} → Class 1, else → Class 0",
                    "benefits": "Better class balance, clearer decision boundary, reduced overfitting, higher accuracy expected",
                    "details": {
                        "threshold": float(threshold),
                        "current_distribution": {
                            str(k): int(v) for k, v in value_counts.items()
                        },
                        "proposed_distribution": {
                            "negative (< threshold)": int(negative_count),
                            "positive (>= threshold)": int(positive_count),
                            "balance": f"{new_balance:.1f}%",
                        },
                    },
                    "example": "Example: Wine quality 7+ = 'Good', else 'Bad'",
                }
            )

    # Add feature reduction recommendation for small datasets
    if total_samples < 200 and n_classes >= 3:
        recommendations.append(
            {
                "type": "feature_reduction",
                "priority": "medium",
                "title": "📊 Consider Reducing TF-IDF Features",
                "message": f"With only {total_samples} samples, 100 TF-IDF features per text column may cause overfitting",
                "suggestion": "Reduce max_features from 100 to 30-50 for text columns",
                "benefits": [
                    "✓ Reduces feature-to-sample ratio",
                    "✓ Focuses on most important words",
                    "✓ Faster training",
                    "✓ Less overfitting",
                ],
            }
        )

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
        "warnings": warnings,
        "recommendations": recommendations,
        "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
        "total_samples": total_samples,
        "n_classes": n_classes,
    }


@app.post("/store-transformation", status_code=status.HTTP_200_OK)
async def store_transformation(
    session_id: str = Form(...),
    transformation_type: str = Form(...),
    threshold: float = Form(None),
):
    """Store transformation parameters to be applied in the notebook"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "target_column" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="Target column not set")

    target_column = sessions[session_id]["target_column"]

    if transformation_type == "binary_transformation":
        if threshold is None:
            raise HTTPException(
                status_code=400, detail="Threshold required for binary transformation"
            )

        # Store transformation parameters in session
        sessions[session_id]["transformation"] = {
            "type": transformation_type,
            "threshold": threshold,
            "target_column": target_column,
        }

        return {
            "message": f"Transformation stored: Values >= {threshold} → Class 1, Values < {threshold} → Class 0",
            "transformation": sessions[session_id]["transformation"],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transformation type: {transformation_type}",
        )


@app.get("/column-info/{session_id}")
async def get_column_info(session_id: str):
    """Get information about all columns for manual review"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    dataset_filename = sessions[session_id]["dataset"]
    file_path = os.path.join(DATA_FOLDER, dataset_filename)
    target_column = sessions[session_id].get("target_column")
    id_columns = sessions[session_id].get("id_columns", [])
    problematic_details = sessions[session_id].get("problematic_column_details", {})

    df = pd.read_csv(file_path)

    # Remove target column from the list
    columns_to_show = [col for col in df.columns if col != target_column]

    column_info = []
    for col in columns_to_show:
        unique_count = df[col].nunique()
        sample_values = df[col].dropna().astype(str).unique()[:5].tolist()

        # Determine the reason for exclusion
        exclusion_reason = None
        if col in id_columns:
            # Find which category this column belongs to
            for category, cols in problematic_details.items():
                if col in cols:
                    exclusion_reason = category.replace("_", " ").title()
                    break
            if not exclusion_reason:
                exclusion_reason = "ID Column"

        column_info.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_count": int(unique_count),
                "sample_values": sample_values,
                "is_auto_excluded": col in id_columns,
                "exclusion_reason": exclusion_reason,
            }
        )

    return {"columns": column_info}


@app.post("/exclude-columns", status_code=status.HTTP_200_OK)
async def exclude_columns(
    session_id: str = Form(...), excluded_columns: str = Form(...)
):
    """Store user-selected columns to exclude from training"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Parse the JSON string of excluded columns
    import json

    try:
        excluded_list = json.loads(excluded_columns)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid excluded_columns format")

    sessions[session_id]["user_excluded_columns"] = excluded_list

    return {
        "message": f"Excluded {len(excluded_list)} column(s) from training",
        "excluded_columns": excluded_list,
    }


@app.post("/model", status_code=status.HTTP_200_OK)
async def set_model(session_id: str = Form(...), model_name: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if "target_column" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="Target column not set")
    sessions[session_id]["model"] = model_name
    return {"message": f"{model_name} set successfully as model."}


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
    # Use relative paths for portability across different machines/users
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
            "SVC(probability=True)",
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

    # Determine which packages to install based on selected model
    base_packages = "pandas numpy scikit-learn ipykernel"
    extra_packages = ""

    if "xgboost" in model_key:
        extra_packages = " xgboost"
    elif "lightgbm" in model_key:
        extra_packages = " lightgbm"

    install_code = f"""%pip install {base_packages}{extra_packages} -q
print("All required packages installed successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(install_code))

    # Import only essential libraries + the specific model chosen by user
    imports_code = f"""# Import required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, classification_report

# Import the selected model
{import_stmt}

print("All libraries imported successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(imports_code))

    # EDA Section
    eda_md = """## 1. Exploratory Data Analysis (EDA)
Let's start by loading and exploring the dataset."""
    nb.cells.append(nbformat.v4.new_markdown_cell(eda_md))

    code1 = """df = pd.read_csv(DATA_PATH)
df.head()"""

    nb.cells.append(nbformat.v4.new_code_cell(code1))

    # Add transformation cell if transformation is stored
    if "transformation" in sessions[session_id]:
        transformation = sessions[session_id]["transformation"]
        if transformation["type"] == "binary_transformation":
            threshold = transformation["threshold"]
            target_col_name = transformation["target_column"]

            transform_md = f"""### 📊 Smart Transformation Applied
Based on class imbalance detection, we're applying a binary transformation to improve model performance:
- **Original column**: `{target_col_name}` (multi-class with imbalance)
- **Transformation**: Values >= {threshold} → Class 1, Values < {threshold} → Class 0
- **Benefits**: Better class balance, clearer decision boundary, reduced overfitting"""
            nb.cells.append(nbformat.v4.new_markdown_cell(transform_md))

            transform_code = f"""# Apply binary transformation to target column
print("Original target distribution:\\n")
print(df['{target_col_name}'].value_counts().sort_index())
# Save original values for reference
df['{target_col_name}_original'] = df['{target_col_name}']
# Apply transformation: >= {threshold} = Class 1, < {threshold} = Class 0
df['{target_col_name}'] = (df['{target_col_name}'] >= {threshold}).astype(int)
print("\\nTransformed target distribution:\\n")
print(df['{target_col_name}'].value_counts().sort_index())
print(f"\\nClass 0 (< {threshold}): {{(df['{target_col_name}'] == 0).sum()}} samples")
print(f"Class 1 (>= {threshold}): {{(df['{target_col_name}'] == 1).sum()}} samples")
print(f"\\nBalance ratio: {{(df['{target_col_name}'] == 0).sum() / (df['{target_col_name}'] == 1).sum() * 100:.1f}}%")"""
            nb.cells.append(nbformat.v4.new_code_cell(transform_code))

    code2 = "# Dataset information\ndf.info()"
    code3 = "# Summary statistics\ndf.describe()"

    # Only show target distribution if no transformation was applied (transformation cell already shows it)
    code4 = None
    if "transformation" not in sessions[session_id]:
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

    # Get ID columns that will be excluded
    id_columns = sessions[session_id].get("id_columns", [])
    id_cols_str = ", ".join([f'"{col}"' for col in id_columns])

    code5 = f"""# Data Quality Assessment
# Columns to be excluded from analysis
excluded_cols = [{id_cols_str}]

# 1. Missing Values
missing_summary = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({{'Missing Count': missing_summary, 'Percentage': missing_pct}})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if not missing_df.empty:
    print("\\n⚠️  Missing Values Detected:")
    print(missing_df)
else:
    print("\\n✓ No missing values found!")

# 2. Duplicate Rows
n_duplicates = df.duplicated().sum()
if n_duplicates > 0:
    print(f"\\n⚠️  Found {{n_duplicates}} duplicate rows ({{n_duplicates/len(df)*100:.1f}}%)")
else:
    print("\\n✓ No duplicate rows")

# 3. Constant Columns (zero variance)
constant_cols = [col for col in df.columns if col not in excluded_cols and df[col].nunique() == 1]
if constant_cols:
    print(f"\\n⚠️  Constant columns (will be dropped): {{', '.join(constant_cols)}}")

# 4. High Cardinality (potential issues) - excluding already detected ID columns
high_card = []
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != '{target_col}' and col not in excluded_cols and df[col].nunique() > 50:
        high_card.append((col, df[col].nunique()))
if high_card:
    print("\\n⚠️  High cardinality columns (may need special handling):")
    for col, n_unique in high_card:
        print(f"    - {{col}}: {{n_unique}} unique values")"""

    # Smart Data Cleaning Strategy
    code6 = f"""impute_report = []
drop_report = []
quality_report = []

# Step 1: Remove duplicate rows
n_duplicates = df.duplicated().sum()
if n_duplicates > 0:
    df = df.drop_duplicates()
    quality_report.append(f"✓ Removed {{n_duplicates}} duplicate rows")

# Step 2: Remove constant columns (zero variance = no predictive power)
constant_cols = [col for col in df.columns if col != '{target_col}' and df[col].nunique() == 1]
if constant_cols:
    df = df.drop(columns=constant_cols)
    drop_report.append(f"❌ Dropped constant columns (zero variance): {{', '.join(constant_cols)}}")

# Step 3: Handle missing values intelligently
for col in df.columns:
    if col == '{target_col}':
        # Drop rows with missing target (can't train on these)
        n_missing_target = df[col].isnull().sum()
        if n_missing_target > 0:
            df = df.dropna(subset=[col])
            quality_report.append(f"⚠️  Dropped {{n_missing_target}} rows with missing target '{{col}}'")
        continue
    
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        missing_pct = (n_missing / len(df)) * 100
        
        # Strategy 1: Drop columns with >50% missing (unreliable)
        if missing_pct > 50:
            df = df.drop(columns=[col])
            drop_report.append(f"❌ Dropped '{{col}}': {{missing_pct:.1f}}% missing (>50% threshold)")
        
        # Strategy 2: Numeric columns
        elif np.issubdtype(df[col].dtype, np.number):
            unique_vals = df[col].dropna().unique()
            
            # Binary numeric (0/1) - Create missing indicator
            if len(unique_vals) <= 2 and set(unique_vals).issubset({{0, 1, 0.0, 1.0}}):
                df[f'{{col}}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(0)
                impute_report.append(f"⚠️  Binary '{{col}}': Created '{{col}}_missing' indicator + filled with 0")
            
            # Continuous numeric - Use median (robust to outliers)
            else:
                median = df[col].median()
                df[col] = df[col].fillna(median)
                impute_report.append(f"✓ Numeric '{{col}}': Filled {{n_missing}} ({{missing_pct:.1f}}%) with median={{median}}")
        
        # Strategy 3: Categorical columns
        else:
            # Low missing (<5%) - Use mode
            if missing_pct < 5:
                mode = df[col].mode().dropna()
                if not mode.empty:
                    fill_value = mode[0]
                    df[col] = df[col].fillna(fill_value)
                    impute_report.append(f"✓ Categorical '{{col}}': Filled {{n_missing}} ({{missing_pct:.1f}}%) with mode='{{fill_value}}'")
                else:
                    df[col] = df[col].fillna('Unknown')
                    impute_report.append(f"⚠️  Categorical '{{col}}': No mode found, filled with 'Unknown'")
            
            # High missing (>=5%) - Create explicit "Missing" category
            else:
                df[col] = df[col].fillna('Missing')
                impute_report.append(f"⚠️  Categorical '{{col}}': Created 'Missing' category ({{missing_pct:.1f}}% missing)")

# Step 4: Handle outliers in numeric columns (optional but recommended)
outlier_report = []
for col in df.select_dtypes(include=[np.number]).columns:
    if col != '{target_col}':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # 3*IQR for extreme outliers only
        upper_bound = Q3 + 3 * IQR
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if n_outliers > 0:
            outlier_pct = n_outliers / len(df) * 100
            if outlier_pct < 5:  # Only cap if <5% (else might be legitimate)
                df[col] = df[col].clip(lower_bound, upper_bound)
                outlier_report.append(f"⚠️  Capped {{n_outliers}} outliers in '{{col}}' ({{outlier_pct:.1f}}%)")

# Print reports
if quality_report:
    for line in quality_report:
        print(line)

if drop_report:
    for line in drop_report:
        print(line)

if impute_report:
    for line in impute_report:
        print(line)

if outlier_report:
    for line in outlier_report:
        print(line)
        
if not quality_report and not drop_report and not impute_report and not outlier_report:
    print('✓ Data is clean! No preprocessing needed.')

print(f"\\n✓ Final dataset shape: {{df.shape}}")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code2))
    nb.cells.append(nbformat.v4.new_code_cell(code3))
    if code4:  # Only add target distribution cell if no transformation was applied
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

    # Determine columns to drop (target + original column if transformation was applied)
    columns_to_drop = [f'"{target_col}"']
    if "transformation" in sessions[session_id]:
        columns_to_drop.append(f'"{target_col}_original"')

    # Build the list of columns to exclude from input_cols
    exclude_from_input = [target_col]
    if "transformation" in sessions[session_id]:
        exclude_from_input.append(f"{target_col}_original")

    # Add user-excluded columns to the exclude list
    user_excluded_columns = sessions[session_id].get("user_excluded_columns", [])
    exclude_from_input.extend(user_excluded_columns)

    exclude_cols_str = ", ".join([f'"{col}"' for col in exclude_from_input])

    # Format user_excluded_columns as a Python list string for the notebook
    user_excluded_str = (
        "[" + ", ".join([f'"{col}"' for col in user_excluded_columns]) + "]"
    )

    code6 = f"""feature_report = []
# Drop target column and original column (if transformation was applied)
columns_to_drop = [{", ".join(columns_to_drop)}]
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
y = df["{target_col}"]

# Encode target variable for classification if it's not numeric
label_encoder = None
if y.dtype == 'object' or y.dtype.name == 'category':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Create clean encoding map with Python ints (not numpy.int64)
    encoding_map = {{str(cls): int(code) for cls, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}}
    feature_report.append(f"Encoded target variable: {{encoding_map}}")
    # Save label encoder
    with open(f'artifacts/label_encoder_{{SESSION_ID}}.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# Identify and drop ID columns (low predictive value)
id_cols = []
for col in X.columns:
    # Check if column name contains 'id' (at word boundaries or with underscores: customer_id, user_id, id, etc.)
    # Avoid false positives like "acidity", "chlorides", "citric acid"
    import re
    if re.search(r'(?:^|_)id(?:$|_)', col.lower()):  # Match _id, id_, _id_, or standalone id
        id_cols.append(col)
    # Check if column has near-unique values (>95% unique = likely an ID)
    # Only for object/string columns to avoid flagging continuous numeric features
    elif X[col].dtype == 'object' and X[col].nunique() / len(X) > 0.95:
        id_cols.append(col)

# User-excluded columns (from manual review)
user_excluded = {user_excluded_str}

# Combine auto-detected IDs with user-excluded columns
all_excluded = list(set(id_cols + user_excluded))

if all_excluded:
    X = X.drop(columns=all_excluded)
    if id_cols:
        feature_report.append(f"Dropped ID columns (auto-detected): {{', '.join(id_cols)}}")
    if user_excluded:
        feature_report.append(f"Dropped columns (user-excluded): {{', '.join(user_excluded)}}")

# Identify text columns (high cardinality) and process with TF-IDF
text_cols = []
cat_cols = []
date_cols = []

for col in X.select_dtypes(include=["object", "category"]).columns:
    # Check if column might be a date
    sample_val = str(X[col].iloc[0])
    if '-' in sample_val and len(sample_val) <= 12:  # Likely a date format like "2024-03-15"
        try:
            pd.to_datetime(X[col].iloc[0])
            date_cols.append(col)
            continue
        except:
            pass
    
    # Not a date, check if text or categorical
    if X[col].nunique() > 50:  # Likely a text column
        text_cols.append(col)
    else:
        cat_cols.append(col)

# Drop date columns (they rarely help with prediction and create noise)
if date_cols:
    X = X.drop(columns=date_cols)
    feature_report.append(f"Dropped date columns (low predictive value): {{', '.join(date_cols)}}")

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
        with open(f"artifacts/tfidf_{{col}}_{{SESSION_ID}}.pkl", 'wb') as f:
            pickle.dump(tfidf, f)
    X = X.drop(columns=text_cols)
    feature_report.append(f"Applied TF-IDF to text columns: {{', '.join(text_cols)}} (100 features each)")

# Save categorical column values for validation in predictions
cat_values = {{}}
for col in cat_cols:
    cat_values[col] = sorted(X[col].unique().tolist())
with open(f'artifacts/categorical_values_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(cat_values, f)

# One-hot encode low-cardinality categoricals
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_report.append(f"One-hot encoded columns: {{', '.join(cat_cols)}}")
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
        feature_report.append(f"Scaled numeric columns: {{', '.join(num_cols)}}")
        # Save scaler for future use
        with open(f'artifacts/scaler_{{SESSION_ID}}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        feature_report.append("No numeric columns to scale.")
else:
    feature_report.append("Skipped scaling (tree-based model)")

# Save feature names for prediction consistency
with open(f'artifacts/features_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# Save input column names (original columns that weren't dropped - what user needs to provide)
# Exclude: target, excluded columns, date columns, ID columns, and auto-generated _missing indicator columns
exclude_cols = [{exclude_cols_str}]
input_cols = [col for col in df.columns 
              if col not in exclude_cols 
              and col not in date_cols 
              and col not in all_excluded
              and not col.endswith('_missing')  # Exclude auto-generated missing indicators
              and not col.endswith('_original')]  # Exclude original target backup
with open(f'artifacts/input_columns_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(input_cols, f)

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
print(f"\\nClassification Report:\\n{{classification_report(y_test, X_test_pred, zero_division=0)}}")

# Save trained model
with open(f'artifacts/model_{{SESSION_ID}}.pkl', 'wb') as f:
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
train_mae = mean_absolute_error(y_train, X_train_pred)
train_mse = mean_squared_error(y_train, X_train_pred)
train_rmse = np.sqrt(train_mse)

# Evaluate on test set
X_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, X_test_pred)
test_mae = mean_absolute_error(y_test, X_test_pred)
test_mse = mean_squared_error(y_test, X_test_pred)
test_rmse = np.sqrt(test_mse)

# Display results
print("TRAINING SET PERFORMANCE")
print(f"R² Score: {{train_r2:.4f}}")
print(f"MAE (Mean Absolute Error): {{train_mae:.4f}}")
print(f"MSE (Mean Squared Error): {{train_mse:.4f}}")
print(f"RMSE (Root Mean Squared Error): {{train_rmse:.4f}}")

print("\\nTEST SET PERFORMANCE")
print(f"R² Score: {{test_r2:.4f}}")
print(f"MAE (Mean Absolute Error): {{test_mae:.4f}}")
print(f"MSE (Mean Squared Error): {{test_mse:.4f}}")
print(f"RMSE (Root Mean Squared Error): {{test_rmse:.4f}}")

# Save trained model
with open(f'artifacts/model_{{SESSION_ID}}.pkl', 'wb') as f:
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
with open(f'artifacts/model_{session_id}.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open(f'artifacts/scaler_{session_id}.pkl', 'rb') as f:
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
        "model": os.path.join(ARTIFACTS_FOLDER, f"model_{session_id}.pkl"),
        "scaler": os.path.join(ARTIFACTS_FOLDER, f"scaler_{session_id}.pkl"),
        "notebook": f"{session_id}.ipynb",
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    def sse_event_stream():
        result_queue = queue.Queue()
        code_cell_counter = [0]  # Use list to make it mutable in closure
        all_cell_results = []  # Store all results for LLM summary

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

            # Store result for LLM summary
            all_cell_results.append(
                {"cell_number": code_cell_counter[0], "result": cell_result}
            )

            # Put result in queue for streaming
            result_queue.put({"cell": code_cell_counter[0], "result": cell_result})

        def execute_notebook():
            """Execute notebook in a separate thread"""
            try:
                nb_to_run = copy.deepcopy(nb)
                # Execute from project root so relative paths work correctly
                # This makes the notebook portable across different machines/users
                project_root = os.path.abspath(".")
                client = NotebookClient(
                    nb_to_run,
                    allow_errors=True,
                    timeout=600,
                    resources={"metadata": {"path": project_root}},
                )

                # Register hook to stream results after each cell
                client.on_cell_executed = on_cell_executed

                # Execute the entire notebook (hook will stream results)
                client.execute(cwd=project_root)

                # Store all results in session for LLM report generation
                sessions[session_id]["cell_results"] = all_cell_results

                # No need to move files - they're already saved directly to artifacts/ folder

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


def _generate_report_background(session_id: str):
    """Background task to generate AI report"""
    try:
        # Get all cell results
        cell_results = sessions[session_id]["cell_results"]
        target_col = sessions[session_id].get("target_column", "unknown")
        model_name = sessions[session_id].get("model", "unknown")
        dataset_name = (
            sessions[session_id]["dataset"].replace(session_id, "").replace(".csv", "")
        )

        # Build context from cell results (filter out HTML and keep only text)
        context_parts = []
        for cell in cell_results:
            cell_num = cell["cell_number"]
            result = cell["result"]

            # Skip HTML results, only include text outputs
            if result and not (isinstance(result, str) and result.startswith("<")):
                # Truncate very long outputs
                result_text = str(result)[:1000] if result else "No output"
                context_parts.append(f"Cell {cell_num} Output:\n{result_text}\n")

        context = "\n".join(context_parts[-10:])  # Use last 10 cells for context

        # Create prompt for LLM
        prompt = f"""You are an expert data scientist analyzing ML pipeline results. Generate a comprehensive, professional report.

Dataset: {dataset_name}
Target Variable: {target_col}
Model Used: {model_name}

Execution Results:
{context}

Please provide a detailed analysis report with the following sections:

1. **Executive Summary**: Brief overview of the ML pipeline execution
2. **Data Quality Assessment**: Comment on dataset size, missing values, and data distribution
3. **Model Performance Analysis**: 
   - Interpret the cross-validation scores
   - Analyze train vs test performance
   - Identify any signs of overfitting or underfitting
4. **Key Findings**: Highlight important insights from the results
5. **Recommendations**: 
   - Suggest improvements for model performance
   - Recommend next steps (feature engineering, hyperparameter tuning, etc.)
6. **Conclusion**: Final assessment and deployment readiness

Format your response in clear sections with bullet points where appropriate. Be specific and actionable."""

        # Generate report using LLM
        response = llm.invoke(prompt)

        # Extract content from response
        report = response.content if hasattr(response, "content") else str(response)

        # Store report in session with status
        sessions[session_id]["ai_report"] = {
            "status": "completed",
            "report": report,
            "generated_at": pd.Timestamp.now().isoformat(),
        }

    except Exception as e:
        import traceback

        # Store error in session
        sessions[session_id]["ai_report"] = {
            "status": "failed",
            "error": f"{str(e)}\n{traceback.format_exc()}",
        }


@app.api_route("/generate/report/{session_id}", methods=["GET", "POST"])
async def generate_report(session_id: str, background_tasks: BackgroundTasks):
    """Start AI-powered report generation in background"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    if "cell_results" not in sessions[session_id]:
        raise HTTPException(
            status_code=404,
            detail="No execution results found. Please run the notebook first.",
        )

    # Check if report is already being generated or completed
    if "ai_report" in sessions[session_id]:
        existing_status = sessions[session_id]["ai_report"].get("status")
        if existing_status == "completed":
            return sessions[session_id]["ai_report"]
        elif existing_status == "generating":
            return {
                "session_id": session_id,
                "status": "generating",
                "message": "Report generation already in progress. Use /report/status/{session_id} to check progress.",
            }

    # Mark report as generating
    sessions[session_id]["ai_report"] = {
        "status": "generating",
        "message": "Report generation in progress...",
    }

    # Start background task
    background_tasks.add_task(_generate_report_background, session_id)

    return {
        "session_id": session_id,
        "status": "generating",
        "message": "Report generation started in background. Use /report/status/{session_id} to check progress.",
    }


@app.get("/report/status/{session_id}")
async def get_report_status(session_id: str):
    """Check the status of report generation"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    if "ai_report" not in sessions[session_id]:
        return {"status": "not_started", "message": "Report not requested yet"}

    return sessions[session_id]["ai_report"]


@app.get("/report/stream/{session_id}")
async def stream_report_status(session_id: str):
    """Stream report generation status via Server-Sent Events"""
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    async def event_generator():
        """Generate SSE events for report status"""
        while True:
            if "ai_report" not in sessions[session_id]:
                yield f"data: {json.dumps({'status': 'not_started'})}\n\n"
                await asyncio.sleep(1)
                continue

            report_data = sessions[session_id]["ai_report"]
            yield f"data: {json.dumps(report_data)}\n\n"

            # Stop streaming if completed or failed
            if report_data.get("status") in ["completed", "failed"]:
                break

            await asyncio.sleep(2)  # Poll every 2 seconds

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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

        # Get original dataset to understand feature types and apply same preprocessing
        df = pd.read_csv(os.path.join(DATA_FOLDER, sessions[session_id]["dataset"]))
        target_col = sessions[session_id]["target_column"]
        X_original = df.drop(columns=[target_col])

        # Convert input_data values to correct types (handle string inputs from frontend)
        typed_input_data = {}
        for col in X_original.columns:
            if col in input_data:
                if pd.api.types.is_numeric_dtype(X_original[col]):
                    # Convert to float/int based on original column type
                    typed_input_data[col] = float(input_data[col])
                else:
                    typed_input_data[col] = str(input_data[col])
            else:
                # Column missing from input - will be filled later
                typed_input_data[col] = None

        # Create a DataFrame with the properly typed input data
        input_df = pd.DataFrame([typed_input_data])

        # Fill missing columns with median/mode from training data
        for col in X_original.columns:
            if col not in input_df.columns:
                if pd.api.types.is_numeric_dtype(X_original[col]):
                    input_df[col] = X_original[col].median()
                else:
                    mode_val = X_original[col].mode()
                    input_df[col] = mode_val[0] if not mode_val.empty else "Unknown"

        # Reorder to match original column order
        input_df = input_df[
            [col for col in X_original.columns if col in input_df.columns]
        ]

        # Apply same preprocessing as training
        # 1. Identify text and categorical columns (same logic as training)
        text_cols = []
        cat_cols = []
        date_cols = []

        for col in input_df.select_dtypes(include=["object", "category"]).columns:
            # Check if column might be a date
            sample_val = str(input_df[col].iloc[0])
            if "-" in sample_val and len(sample_val) <= 12:
                try:
                    pd.to_datetime(input_df[col].iloc[0])
                    date_cols.append(col)
                    continue
                except Exception:
                    pass

            # Not a date, check if text or categorical
            if X_original[col].nunique() > 50:
                text_cols.append(col)
            else:
                cat_cols.append(col)

        # Drop date columns
        if date_cols:
            input_df = input_df.drop(columns=date_cols)

        # 2. Process text columns with TF-IDF
        tfidf_features = []
        if text_cols:
            for col in text_cols:
                tfidf_file = os.path.join(
                    ARTIFACTS_FOLDER, f"tfidf_{col}_{session_id}.pkl"
                )
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

        # 5. Scale numeric features BEFORE aligning (same order as training)
        if scaler:
            num_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                input_df[num_cols] = scaler.transform(input_df[num_cols])

        # 6. Align columns with training data (after scaling)
        features_file = os.path.join(ARTIFACTS_FOLDER, f"features_{session_id}.pkl")
        if os.path.exists(features_file):
            with open(features_file, "rb") as f:
                expected_features = pickle.load(f)

            # Add missing columns with 0
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Remove extra columns and reorder to match training
            input_df = input_df[expected_features]
        elif hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_features]

        # Convert to array
        X = input_df.values

        # Predict
        prediction = model.predict(X)[0]

        # Load label encoder if exists (for classification with string targets)
        label_encoder_file = os.path.join(
            ARTIFACTS_FOLDER, f"label_encoder_{session_id}.pkl"
        )
        label_encoder = None
        if os.path.exists(label_encoder_file):
            with open(label_encoder_file, "rb") as f:
                label_encoder = pickle.load(f)

        # Get probability for classifiers if available
        probability = None
        has_predict_proba = hasattr(model, "predict_proba")
        try:
            if has_predict_proba:
                proba = model.predict_proba(X)[0]
                # If label encoder exists, use original class names
                if label_encoder:
                    probability = {
                        str(label): float(prob)
                        for label, prob in zip(label_encoder.classes_, proba)
                    }
                else:
                    probability = {str(int(i)): float(p) for i, p in enumerate(proba)}
        except Exception as prob_error:
            probability = {"error": str(prob_error), "has_method": has_predict_proba}

        # Decode prediction if label encoder exists
        if label_encoder:
            try:
                prediction_value = label_encoder.inverse_transform([int(prediction)])[0]
            except Exception:
                prediction_value = str(prediction)
        # Convert prediction to appropriate type (string for classification, float for regression)
        elif has_predict_proba:
            # Classification: keep as string
            prediction_value = str(prediction)
        else:
            # Regression: convert to float
            try:
                prediction_value = float(prediction)
            except (ValueError, TypeError):
                prediction_value = str(prediction)

        return {
            "prediction": prediction_value,
            "probability": probability if probability is not None else "null_value",
            "features_used": input_df.columns.tolist(),
            "input_shape": list(X.shape),
            "scaled_values": X[0].tolist()[:5],  # First 5 values for debugging
            "original_input": list(input_data.values())[:5],  # Original values
            "scaler_loaded": scaler is not None,
            "has_predict_proba": has_predict_proba,
            "label_encoded": label_encoder is not None,
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

        # Delete all files in artifacts folder
        for file in os.listdir(ARTIFACTS_FOLDER):
            file_path = os.path.join(ARTIFACTS_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_deleted.append(file_path)

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


@app.get("/input-columns/{session_id}")
async def get_input_columns(session_id: str):
    """Get the list of input columns needed for prediction (original columns that weren't dropped)"""
    import pickle

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    input_cols_file = os.path.join(ARTIFACTS_FOLDER, f"input_columns_{session_id}.pkl")
    if not os.path.exists(input_cols_file):
        raise HTTPException(
            status_code=404,
            detail="Input columns file not found. Train the model first.",
        )

    try:
        with open(input_cols_file, "rb") as f:
            input_columns = pickle.load(f)

        target_column = sessions[session_id].get("target_column", "")

        return {
            "input_columns": input_columns,
            "count": len(input_columns),
            "target_column": target_column,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load input columns: {str(e)}"
        )


@app.get("/categorical-values/{session_id}")
async def get_categorical_values(session_id: str):
    """Get the valid categorical values for each column (for dropdowns in prediction form)"""
    import pickle

    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    cat_values_file = os.path.join(
        ARTIFACTS_FOLDER, f"categorical_values_{session_id}.pkl"
    )
    if not os.path.exists(cat_values_file):
        # Return empty dict if file doesn't exist (no categorical columns)
        return {"categorical_values": {}}

    try:
        with open(cat_values_file, "rb") as f:
            categorical_values = pickle.load(f)

        return {"categorical_values": categorical_values}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load categorical values: {str(e)}"
        )


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

        # Delete any TF-IDF vectorizer and feature files from artifacts folder
        for file in os.listdir(ARTIFACTS_FOLDER):
            if (
                file.startswith("tfidf_") or file.startswith("features_")
            ) and file.endswith(f"_{session_id}.pkl"):
                file_path = os.path.join(ARTIFACTS_FOLDER, file)
                os.remove(file_path)
                files_deleted.append(file_path)

        # Remove session from memory
        del sessions[session_id]

        return {
            "message": "Session cleaned up successfully",
            "files_deleted": files_deleted,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
