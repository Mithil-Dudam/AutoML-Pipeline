import os
import uuid
import pandas as pd
import re
import nbformat
from nbclient import NotebookClient
import copy
import json
import queue
import threading
import asyncio
import logging
from datetime import datetime
import chardet
import csv
from dotenv import load_dotenv
import pickle

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    status,
    HTTPException,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from langchain_ollama import ChatOllama
from session_manager import get_session_manager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Dict, Any

load_dotenv()

DATA_FOLDER = "./dataset"
NOTEBOOKS_FOLDER = "./notebooks"
ARTIFACTS_FOLDER = "./artifacts"
llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://ollama:11434")

# Create necessary directories
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(NOTEBOOKS_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

# Configure structured logging (console only - Docker captures stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Add custom filter for request IDs
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True


logger.addFilter(RequestIDFilter())

app = FastAPI()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to add request IDs for logging
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    try:
        response = await call_next(request)
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}",
            extra={"request_id": request_id},
        )
        return response
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
            extra={"request_id": request_id},
            exc_info=True,
        )
        raise


# Initialize Redis session manager
session_manager = get_session_manager()


# Helper function to safely read dataset
def safe_read_dataset(session_id: str) -> pd.DataFrame:
    """Safely read dataset from session, with error handling."""
    dataset_filename = session_manager.get_field(session_id, "dataset")
    if not dataset_filename:
        raise HTTPException(status_code=400, detail="Dataset not found in session")

    dataset_path = os.path.join(DATA_FOLDER, dataset_filename)
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found on server. Session may have expired.",
        )

    try:
        df = pd.read_csv(dataset_path)
        if df.empty:
            raise HTTPException(
                status_code=500, detail="Dataset file is empty or corrupted"
            )
        return df
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=500, detail="Dataset file is corrupted and cannot be read"
        )
    except Exception as e:
        logger.error(f"Error reading dataset for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


# Background cleanup task
async def cleanup_expired_sessions():
    """Background task to clean up files for expired sessions."""
    while True:
        await asyncio.sleep(600)  # Run every 10 minutes
        try:
            all_sessions = session_manager.list_all_sessions()
            for session_id in all_sessions:
                ttl = session_manager.get_ttl(session_id)

                # If session expires in less than 5 minutes or already expired
                if ttl and ttl < 300:
                    session_data = session_manager.get(session_id)
                    if session_data:
                        # Delete dataset file
                        if "dataset" in session_data:
                            dataset_path = os.path.join(
                                DATA_FOLDER, session_data["dataset"]
                            )
                            if os.path.exists(dataset_path):
                                os.remove(dataset_path)

                        # Delete notebook file
                        notebook_path = os.path.join(
                            NOTEBOOKS_FOLDER, f"{session_id}.ipynb"
                        )
                        if os.path.exists(notebook_path):
                            os.remove(notebook_path)

                        # Delete model artifacts
                        if "artifacts" in session_data:
                            for artifact in session_data["artifacts"]:
                                artifact_path = os.path.join(ARTIFACTS_FOLDER, artifact)
                                if os.path.exists(artifact_path):
                                    os.remove(artifact_path)
        except Exception as e:
            # Log error but don't crash the background task
            logger.error(f"Cleanup task error: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """Start background cleanup task on app startup."""
    asyncio.create_task(cleanup_expired_sessions())

    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    auth_status = "ENABLED" if REQUIRE_AUTH else "DISABLED"
    logger.info(f"Application started successfully - Authentication: {auth_status}")

    logger.info("Application started successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/dataset", status_code=status.HTTP_201_CREATED)
@limiter.limit("100/minute")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    # Validate file is a CSV
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Validate file size (max 500MB for better support of larger datasets)
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(file_content) > MAX_FILE_SIZE:
        file_size_mb = len(file_content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 500MB. Consider reducing your dataset size by sampling rows or removing unnecessary columns.",
        )

    session_id = str(uuid.uuid4())
    filename_base = (
        file.filename.rsplit(".", 1)[0] if "." in file.filename else file.filename
    )
    file_location = os.path.join(DATA_FOLDER, filename_base + session_id + ".csv")

    # Detect encoding
    detected = chardet.detect(file_content)
    encoding = detected["encoding"] if detected["encoding"] else "utf-8"

    # Fallback to common encodings if detection fails
    if encoding.lower() not in [
        "utf-8",
        "utf-16",
        "ascii",
        "latin-1",
        "iso-8859-1",
        "cp1252",
    ]:
        encoding = "utf-8"

    # Save file
    with open(file_location, "wb") as buffer:
        buffer.write(file_content)

    # Detect delimiter by sampling first few lines
    try:
        with open(file_location, "r", encoding=encoding) as f:
            sample = f.read(8192)  # Read first 8KB
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
    except Exception:
        delimiter = ","  # Default to comma

    # Try to read and validate CSV with detected encoding and delimiter
    try:
        df = pd.read_csv(file_location, encoding=encoding, delimiter=delimiter)
    except pd.errors.EmptyDataError:
        os.remove(file_location)
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        os.remove(file_location)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(e)}. Detected encoding: {encoding}, delimiter: '{delimiter}'",
        )
    except UnicodeDecodeError:
        # Try alternative encodings
        for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                df = pd.read_csv(
                    file_location, encoding=alt_encoding, delimiter=delimiter
                )
                encoding = alt_encoding
                break
            except Exception:
                continue
        else:
            os.remove(file_location)
            raise HTTPException(
                status_code=400,
                detail=f"Unable to decode file. Tried encodings: {encoding}, latin-1, cp1252, iso-8859-1",
            )
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

    # Check for duplicate column names
    columns = df.columns.tolist()
    if len(columns) != len(set(columns)):
        os.remove(file_location)
        duplicates = [col for col in columns if columns.count(col) > 1]
        unique_duplicates = list(set(duplicates))
        raise HTTPException(
            status_code=400,
            detail=f"CSV file contains duplicate column names: {', '.join(unique_duplicates)}. Please ensure all column names are unique.",
        )

    # Check for problematic column names (contains special characters that break code generation)
    problematic_chars = ["\\", '"', "'", "\n", "\r", "\t"]
    problematic_columns = []
    for col in columns:
        if any(char in str(col) for char in problematic_chars):
            problematic_columns.append(col)

    if problematic_columns:
        os.remove(file_location)
        raise HTTPException(
            status_code=400,
            detail=f"Column names contain invalid characters (quotes, backslashes, newlines): {', '.join(problematic_columns)}. Please rename these columns and re-upload.",
        )

    # Store session in Redis ONLY after all validations pass
    session_manager.create(
        session_id,
        {
            "dataset": filename_base + session_id + ".csv",
            "columns": columns,
        },
    )

    logger.info(
        f"Dataset uploaded successfully: {len(df)} rows, {len(columns)} columns",
        extra={"request_id": request.state.request_id},
    )

    return {"columns": columns, "session_id": session_id}


@app.post("/target-column", status_code=status.HTTP_200_OK)
async def set_target_column(session_id: str = Form(...), column_name: str = Form(...)):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Read dataset with error handling
    df = safe_read_dataset(session_id)

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

    # Validate column has more than one unique value
    n_unique = df[column_name].nunique()
    if n_unique < 2:
        unique_val = df[column_name].iloc[0] if len(df) > 0 else "N/A"
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column_name}' has only 1 unique value (all rows = {unique_val}). Cannot train a model - target must have at least 2 different values.",
        )

    session_manager.set_field(session_id, "target_column", column_name)
    session_manager.set_field(session_id, "dataframe", df)  # Store for later use

    # Analyze target distribution for warnings and recommendations
    warnings = []
    recommendations = []

    # Smart Column Detection - Detect columns that should be dropped
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
        # Don't flag numeric columns with high uniqueness (could be useful features)
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
        session_manager.set_field(session_id, "id_columns", all_problematic)
        session_manager.set_field(
            session_id, "problematic_column_details", problematic_columns
        )

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
        # Determine recommended strategy based on data characteristics
        if min_class_count < 100:
            recommended_strategy = "class_weights"
            strategy_reason = (
                "Small minority class - avoid oversampling to prevent overfitting"
            )
        elif max_class_count > 50000:
            recommended_strategy = "undersample"
            strategy_reason = "Large dataset - safe to undersample majority class"
        elif imbalance_ratio > 100:
            recommended_strategy = "combined"
            strategy_reason = "Extreme imbalance - combine techniques for best results"
        else:
            recommended_strategy = "class_weights"
            strategy_reason = (
                "Moderate imbalance - class weights are effective and efficient"
            )

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
                "recommended_strategy": recommended_strategy,
                "strategy_reason": strategy_reason,
                "available_strategies": {
                    "none": "No resampling (not recommended for severe imbalance)",
                    "class_weights": "Use class weights in model (works with all data, no resampling)",
                    "undersample": "Random undersample majority class (fast, works well for large datasets)",
                    "oversample": "Random oversample minority class (keeps all data, risk of overfitting)",
                    "combined": "Undersample majority + class weights (best for extreme imbalance)",
                },
            }
        )

    # Generate recommendations for binary transformation
    if n_classes >= 3 and pd.api.types.is_numeric_dtype(df[column_name]):
        # Suggest binary transformation if conditions are met
        should_suggest = (
            (
                len(minority_classes) > 0 and n_classes >= 3
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

    # Store warnings and recommendations in session for notebook generation
    session_manager.set_field(session_id, "warnings", warnings)
    session_manager.set_field(session_id, "recommendations", recommendations)

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
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "target_column") is None:
        raise HTTPException(status_code=400, detail="Target column not set")

    target_column = session_manager.get_field(session_id, "target_column")

    if transformation_type == "binary_transformation":
        if threshold is None:
            raise HTTPException(
                status_code=400, detail="Threshold required for binary transformation"
            )

        # Store transformation parameters in session
        session_manager.set_field(
            session_id,
            "transformation",
            {
                "type": transformation_type,
                "threshold": threshold,
                "target_column": target_column,
            },
        )

        transformation = session_manager.get_field(session_id, "transformation")
        return {
            "message": f"Transformation stored: Values >= {threshold} → Class 1, Values < {threshold} → Class 0",
            "transformation": transformation,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transformation type: {transformation_type}",
        )


@app.get("/column-info/{session_id}")
async def get_column_info(session_id: str):
    """Get information about all columns for manual review"""
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    df = safe_read_dataset(session_id)

    target_column = session_manager.get_field(session_id, "target_column")
    id_columns = session_manager.get_field(session_id, "id_columns", [])
    problematic_details = session_manager.get_field(
        session_id, "problematic_column_details", {}
    )

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
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Parse the JSON string of excluded columns
    try:
        excluded_list = json.loads(excluded_columns)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid excluded_columns format")

    # Validate we're not excluding ALL columns
    target_column = session_manager.get_field(session_id, "target_column")

    try:
        df = safe_read_dataset(session_id)
        all_columns = set(df.columns)

        # Get problematic columns that will be auto-excluded
        id_columns = session_manager.get_field(session_id, "id_columns", [])

        # Calculate remaining feature columns
        excluded_set = set(excluded_list) | set(id_columns) | {target_column}
        remaining_features = all_columns - excluded_set

        if len(remaining_features) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot exclude all columns. You're excluding {len(excluded_list)} columns, {len(id_columns)} are auto-excluded, and 1 is the target. This leaves 0 features for training. Please keep at least 1 feature column.",
            )

        if len(remaining_features) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(remaining_features)} feature column(s) remaining. For better model performance, keep at least 2 feature columns.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating excluded columns: {e}")
        # Don't fail if validation has issues, but log it

    session_manager.set_field(session_id, "user_excluded_columns", excluded_list)

    return {
        "message": f"Excluded {len(excluded_list)} column(s) from training",
        "excluded_columns": excluded_list,
    }


@app.post("/model", status_code=status.HTTP_200_OK)
@limiter.limit("100/minute")
async def set_model(
    request: Request, session_id: str = Form(...), model_name: str = Form(...)
):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "target_column") is None:
        raise HTTPException(status_code=400, detail="Target column not set")
    session_manager.set_field(session_id, "model", model_name)
    return {"message": f"{model_name} set successfully as model."}


@app.post("/imbalance-strategy", status_code=status.HTTP_200_OK)
async def set_imbalance_strategy(
    session_id: str = Form(...), strategy: str = Form(...)
):
    """Set the strategy for handling class imbalance.

    Available strategies:
    - none: No resampling
    - class_weights: Use class weights in model (default for most cases)
    - undersample: Random undersample majority class
    - oversample: Random oversample minority class
    - combined: Undersample + class weights
    """
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    valid_strategies = [
        "none",
        "class_weights",
        "undersample",
        "oversample",
        "combined",
    ]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}",
        )

    session_manager.set_field(session_id, "imbalance_strategy", strategy)
    return {
        "message": f"Imbalance handling strategy set to: {strategy}",
        "strategy": strategy,
    }


@app.get("/generate/notebook")
async def generate_notebook(session_id: str):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if (
        session_manager.get_field(session_id, "target_column") is None
        or session_manager.get_field(session_id, "model") is None
    ):
        raise HTTPException(status_code=400, detail="Target column or model not set")

    # Enforce imbalance strategy selection for severely imbalanced datasets
    warnings = session_manager.get_field(session_id, "warnings", [])
    has_severe_imbalance = any(w.get("type") == "severe_imbalance" for w in warnings)

    if has_severe_imbalance:
        strategy = session_manager.get_field(session_id, "imbalance_strategy")
        if not strategy:
            raise HTTPException(
                status_code=400,
                detail="Severe class imbalance detected. Please select an imbalance handling strategy before training. Go back and choose: oversample, undersample, class_weights, or combined.",
            )

    notebook_path = os.path.join(NOTEBOOKS_FOLDER, f"{session_id}.ipynb")
    nb = nbformat.v4.new_notebook()

    # Add title markdown cell
    target_col = session_manager.get_field(session_id, "target_column", None)
    model_name = session_manager.get_field(session_id, "model", "")
    dataset_filename = session_manager.get_field(session_id, "dataset", "")
    dataset_name = dataset_filename.replace(session_id, "").replace(".csv", "")

    # Build cell titles list based on configuration (for frontend mapping)
    cell_titles = [
        "Configuration",
        "Package Installation",
        "Library Imports",
        "First 5 Rows of Dataset",
    ]

    # Add transformation cell title if transformation is applied
    if session_manager.get_field(session_id, "transformation") is not None:
        cell_titles.append("Binary Transformation")

    cell_titles.extend(["Dataset Info", "Summary Statistics"])

    # Add target distribution cell only if no transformation (transformation cell shows it)
    if session_manager.get_field(session_id, "transformation") is None:
        cell_titles.append("Target Column Distribution")

    cell_titles.extend(
        [
            "Missing Values Check",
            "Missing Value Imputation",
            "Data Preparation",
            "Train-Test Split (Before Feature Engineering)",
            "Feature Engineering (No Data Leakage)",
        ]
    )

    # Add class imbalance handling cell title if strategy is set
    strategy = session_manager.get_field(session_id, "imbalance_strategy", "")
    if strategy and strategy != "none":
        cell_titles.append("Class Imbalance Handling")

    cell_titles.extend(
        ["K-Fold Cross-Validation", "Final Model Training and Evaluation"]
    )

    # Store cell titles in session for SSE streaming
    session_manager.set_field(session_id, "cell_titles", cell_titles)

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
    dataset_filename = session_manager.get_field(session_id, "dataset", "")
    # Compute the path outside the f-string to avoid backslash issues
    data_path = os.path.join(DATA_FOLDER, dataset_filename).replace("\\", "/")
    config_code = f"""# Configuration
# Update these paths if you move the notebook to a different location
SESSION_ID = "{session_id}"
DATA_PATH = r'{data_path}'
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
            "XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, eval_metric='logloss', verbosity=0, random_state=42)",
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
            "XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, verbosity=0, random_state=42)",
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

    model_key = session_manager.get_field(session_id, "model", "").strip().lower()
    import_stmt, model_ctor, scoring = model_imports.get(
        model_key,
        (
            "from sklearn.linear_model import LogisticRegression",
            "LogisticRegression()",
            "accuracy" if is_classifier else "r2",
        ),
    )

    # Modify model constructor to add class_weight if using class weights strategy
    strategy = session_manager.get_field(session_id, "imbalance_strategy", "")
    if is_classifier and strategy in ["class_weights", "combined"]:
        # Check which models support class_weight parameter
        models_with_class_weight = [
            "logistic regression",
            "decision tree classifier",
            "random forest classifier",
            "support vector machine",
            "ridge classifier",
            "linear discriminant",
            "perceptron",
        ]

        if any(m in model_key for m in models_with_class_weight):
            # Add class_weight='balanced' to model constructor
            if model_ctor.endswith("()"):
                model_ctor = model_ctor[:-1] + "class_weight='balanced')"
            elif model_ctor.endswith(")"):
                model_ctor = model_ctor[:-1] + ", class_weight='balanced')"

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
    if session_manager.get_field(session_id, "transformation") is not None:
        transformation = session_manager.get_field(session_id, "transformation")
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
    if session_manager.get_field(session_id, "transformation") is None:
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
    id_columns = session_manager.get_field(session_id, "id_columns", [])
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
        print(f"    - {{col}}: {{n_unique}} unique values")

# 5. Outlier Detection (IQR method)
outlier_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if col != '{target_col}' and col not in excluded_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # Avoid division by zero for constant columns
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df)) * 100
                outlier_cols.append((col, outliers, outlier_pct))

if outlier_cols:
    print("\\n📊 Outlier Detection (IQR Method):")
    print("Note: Outliers are NOT removed - they may be important for your use case!")
    for col, count, pct in outlier_cols:
        print(f"  {{col}}: {{count}} outliers ({{pct:.1f}}%)")
    print("\\n💡 Keeping outliers is correct for:")
    print("  • Fraud detection (outliers = fraud cases)")
    print("  • Medical diagnosis (outliers = rare conditions)")
    print("  • Anomaly detection (outliers = what you're looking for!)")"""

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
        
if not quality_report and not drop_report and not impute_report:
    print('✓ Excellent! Dataset is clean and ready for modeling.')
    print('  • No missing values detected')
    print('  • No duplicate rows found')
    print('  • No constant columns to remove')

print(f"\\n✓ Final dataset shape: {{df.shape}}")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code2))
    nb.cells.append(nbformat.v4.new_code_cell(code3))
    if code4:  # Only add target distribution cell if no transformation was applied
        nb.cells.append(nbformat.v4.new_code_cell(code4))
    nb.cells.append(nbformat.v4.new_code_cell(code5))

    # Data Cleaning Section
    cleaning_md = """## 2. Data Cleaning & Missing Value Imputation
Handle missing values by imputing with appropriate strategies."""
    nb.cells.append(nbformat.v4.new_markdown_cell(cleaning_md))
    nb.cells.append(nbformat.v4.new_code_cell(code6))

    # Feature engineering: one-hot encode categoricals (except target), scale numerics if needed
    model = session_manager.get_field(session_id, "model", "").lower()
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
    if session_manager.get_field(session_id, "transformation") is not None:
        columns_to_drop.append(f'"{target_col}_original"')

    # Build the list of columns to exclude from input_cols
    exclude_from_input = [target_col]
    if session_manager.get_field(session_id, "transformation") is not None:
        exclude_from_input.append(f"{target_col}_original")

    # Add user-excluded columns to the exclude list
    user_excluded_columns = session_manager.get_field(
        session_id, "user_excluded_columns", []
    )
    exclude_from_input.extend(user_excluded_columns)

    exclude_cols_str = ", ".join([f'"{col}"' for col in exclude_from_input])

    # Format user_excluded_columns as a Python list string for the notebook
    user_excluded_str = (
        "[" + ", ".join([f'"{col}"' for col in user_excluded_columns]) + "]"
    )

    # NEW FLOW: Prepare data, split FIRST, then do feature engineering
    # Step 1: Prep - Separate X/y, drop IDs, identify column types
    code_prep = f"""# Step 1: Separate features and target
columns_to_drop = [{", ".join(columns_to_drop)}]
X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
y = df["{target_col}"]

# Encode target variable for classification if it's not numeric
label_encoder = None
if y.dtype == 'object' or y.dtype.name == 'category':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    encoding_map = {{str(cls): int(code) for cls, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}}
    print(f"Encoded target variable: {{encoding_map}}")
    with open(f'artifacts/label_encoder_{{SESSION_ID}}.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# Step 2: Identify and drop ID columns
id_cols = []
import re
for col in X.columns:
    if re.search(r'(?:^|_)id(?:$|_)', col.lower()):
        id_cols.append(col)
    elif X[col].dtype == 'object' and X[col].nunique() / len(X) > 0.95:
        id_cols.append(col)

user_excluded = {user_excluded_str}
all_excluded = list(set(id_cols + user_excluded))

if all_excluded:
    X = X.drop(columns=all_excluded)
    if id_cols:
        print(f"Dropped ID columns (auto-detected): {{', '.join(id_cols)}}")
    if user_excluded:
        print(f"Dropped columns (user-excluded): {{', '.join(user_excluded)}}")

# Step 3: Identify column types
text_cols = []
cat_cols = []
date_cols = []

for col in X.select_dtypes(include=["object", "category"]).columns:
    sample_vals = X[col].dropna().head(10)
    if len(sample_vals) > 0:
        date_count = 0
        for val in sample_vals:
            try:
                pd.to_datetime(val)
                date_count += 1
            except:
                pass
        if date_count / len(sample_vals) > 0.5:
            date_cols.append(col)
            continue
    
    unique_ratio = X[col].nunique() / len(X)
    avg_str_length = X[col].astype(str).str.len().mean()
    
    if (unique_ratio > 0.5 and X[col].nunique() > 50) or avg_str_length > 50:
        text_cols.append(col)
    else:
        cat_cols.append(col)

if date_cols:
    X = X.drop(columns=date_cols)
    print(f"Dropped date columns: {{', '.join(date_cols)}}")

# Get numeric columns (exclude text/categorical)
numeric_cols = [col for col in X.columns if col not in text_cols and col not in cat_cols]

print(f"\\nColumn types detected:")
if numeric_cols:
    print(f"  Numeric ({{len(numeric_cols)}}): {{', '.join(numeric_cols)}}")
if text_cols:
    print(f"  Text ({{len(text_cols)}}): {{', '.join(text_cols)}}")
if cat_cols:
    print(f"  Categorical ({{len(cat_cols)}}): {{', '.join(cat_cols)}}")
if not text_cols and not cat_cols:
    print(f"  ✓ All columns are numeric - no text or categorical encoding needed")
    print(f"  ✓ Dataset has {{len(X.columns)}} numeric features ready for modeling")

# Feature count warnings
warnings = []
if len(text_cols) > 10:
    warnings.append(f"⚠️  HIGH TEXT COLUMN COUNT: {{len(text_cols)}} text columns detected!")
    warnings.append(f"   Each text column creates 100 TF-IDF features = {{len(text_cols) * 100}} features total")
    warnings.append(f"   This may cause memory issues. Consider: (1) Drop less important text columns, or (2) Reduce max_features")
    
estimated_features = len(X.columns) - len(text_cols) + (len(text_cols) * 100)
if estimated_features > 500:
    warnings.append(f"⚠️  VERY HIGH FEATURE COUNT: ~{{estimated_features}} features expected after encoding!")
    warnings.append(f"   Risk: Memory issues, slow training, overfitting")
    warnings.append(f"   Recommendation: Use feature selection or dimensionality reduction")

if warnings:
    print("\\n")
    print("FEATURE COUNT WARNINGS:")
    for w in warnings:
        print(w)

# Save input columns for predictions
exclude_cols = [{exclude_cols_str}]
input_cols = [col for col in df.columns 
              if col not in exclude_cols and col not in date_cols and col not in all_excluded
              and not col.endswith('_missing') and not col.endswith('_original')]
with open(f'artifacts/input_columns_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(input_cols, f)
"""

    # Step 2: Split code
    code_split = (
        """# CRITICAL: Split BEFORE feature engineering to prevent data leakage!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42"""
        + (", stratify=y" if is_classifier else "")
        + """)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
"""
    )

    # Step 3: Feature engineering (fit on train only)
    code_features = f"""feature_report = []

# Text columns: TF-IDF (FIT ON TRAIN ONLY!)
tfidf_features_train = []
tfidf_features_test = []

# Check for too many text columns
if len(text_cols) > 10:
    print("\\n")
    print("⚠️  WARNING: HIGH TEXT COLUMN COUNT!")
    print(f"Detected {{len(text_cols)}} text columns.")
    print(f"Each will create 100 TF-IDF features = {{len(text_cols) * 100}} text features!")
    print("This may cause:")
    print("  • High memory usage")
    print("  • Slow training times")
    print("  • Potential overfitting")
    print("\\nConsider dropping less important text columns before training.\\n")

if text_cols:
    for col in text_cols:
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix_train = tfidf.fit_transform(X_train[col].fillna('').astype(str))
        tfidf_matrix_test = tfidf.transform(X_test[col].fillna('').astype(str))
        
        tfidf_df_train = pd.DataFrame(
            tfidf_matrix_train.toarray(), 
            columns=[f"{{col}}_tfidf_{{i}}" for i in range(tfidf_matrix_train.shape[1])],
            index=X_train.index
        )
        tfidf_df_test = pd.DataFrame(
            tfidf_matrix_test.toarray(), 
            columns=[f"{{col}}_tfidf_{{i}}" for i in range(tfidf_matrix_test.shape[1])],
            index=X_test.index
        )
        
        tfidf_features_train.append(tfidf_df_train)
        tfidf_features_test.append(tfidf_df_test)
        
        with open(f"artifacts/tfidf_{{col}}_{{SESSION_ID}}.pkl", 'wb') as f:
            pickle.dump(tfidf, f)
    
    X_train = X_train.drop(columns=text_cols)
    X_test = X_test.drop(columns=text_cols)
    feature_report.append(f"TF-IDF on text columns: {{', '.join(text_cols)}}")

# Categorical columns: split by cardinality
cat_values = {{}}
low_card_cats = []
high_card_cats = []

for col in cat_cols:
    cat_values[col] = sorted(X_train[col].unique().tolist())
    n_unique = X_train[col].nunique()
    if n_unique <= 20:
        low_card_cats.append(col)
    else:
        high_card_cats.append(col)

with open(f'artifacts/categorical_values_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(cat_values, f)

# One-hot encoding (FIT ON TRAIN ONLY!)
if low_card_cats:
    X_train = pd.get_dummies(X_train, columns=low_card_cats, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=low_card_cats, drop_first=True)
    
    # Align test with train (add missing, remove extra, reorder)
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[[col for col in X_train.columns if col in X_test.columns]]
    
    feature_report.append(f"One-hot encoded: {{', '.join(low_card_cats)}}")

# Label encoding (FIT ON TRAIN ONLY!)
if high_card_cats:
    for col in high_card_cats:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test[col] = le.transform(X_test[col])
        
        with open(f'artifacts/le_{{col}}_{{SESSION_ID}}.pkl', 'wb') as f:
            pickle.dump(le, f)
    feature_report.append(f"Label encoded: {{', '.join(high_card_cats)}}")

# Combine TF-IDF features
if tfidf_features_train:
    X_train = pd.concat([X_train] + tfidf_features_train, axis=1)
    X_test = pd.concat([X_test] + tfidf_features_test, axis=1)

# Scaling (FIT ON TRAIN ONLY!)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
if {str(needs_scaling)}:
    if num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        feature_report.append(f"Scaled {{len(num_cols)}} numeric columns")
        
        with open(f'artifacts/scaler_{{SESSION_ID}}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
else:
    feature_report.append("Skipped scaling (tree-based model)")

# Save final feature names
with open(f'artifacts/features_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(list(X_train.columns), f)

print("\\n✅ Feature Engineering Complete (NO DATA LEAKAGE!):")
for line in feature_report:
    print(f"  {{line}}")
print(f"\\nFinal: Train {{X_train.shape}}, Test {{X_test.shape}}")

# Final feature count check
n_features = X_train.shape[1]
if n_features > 500:
    print("\\n")
    print("⚠️  WARNING: VERY HIGH FEATURE COUNT!")
    print(f"Final feature count: {{n_features}}")
    print("Risks:")
    print("  • Memory issues during training")
    print("  • Overfitting (model memorizes training data)")
    print("  • Slow predictions in production")
    print("Recommendations:")
    print("  • Use feature selection (e.g., SelectKBest, RFE)")
    print("  • Apply PCA for dimensionality reduction")
    print("  • Consider simpler models (tree-based handle high dims better)")
elif n_features > 200:
    print(f"\\n💡 Feature count ({{n_features}}) is high but manageable. Monitor training time and memory.")
"""

    # Add cells in the correct order: prep → split → feature engineering
    prep_md = """## 3. Data Preparation
Separate features from target, drop ID columns, identify column types."""
    nb.cells.append(nbformat.v4.new_markdown_cell(prep_md))
    nb.cells.append(nbformat.v4.new_code_cell(code_prep))

    split_md = """## 4. Train-Test Split (Before Feature Engineering)
**CRITICAL**: Split data BEFORE preprocessing to prevent test data from leaking into training transformers!

All transformers (TF-IDF, scaler, encoders) will be fitted ONLY on training data."""
    nb.cells.append(nbformat.v4.new_markdown_cell(split_md))
    nb.cells.append(nbformat.v4.new_code_cell(code_split))

    feature_md = """## 5. Feature Engineering (No Data Leakage!)
Transform features using ONLY training data:
- TF-IDF: Fitted on train, applied to both
- Encoders: Fitted on train, applied to both  
- Scaler: Fitted on train, applied to both"""
    nb.cells.append(nbformat.v4.new_markdown_cell(feature_md))
    nb.cells.append(nbformat.v4.new_code_cell(code_features))

    # Add class imbalance handling for classification
    if is_classifier:
        # Check if there's severe imbalance in the session warnings
        warnings = session_manager.get_field(session_id, "warnings", [])
        imbalance_warning = next(
            (w for w in warnings if w.get("type") == "severe_imbalance"), None
        )

        if imbalance_warning:
            # Get user-selected strategy or use recommended default
            strategy = session_manager.get_field(session_id, "imbalance_strategy")
            if not strategy:
                strategy = imbalance_warning.get(
                    "recommended_strategy", "class_weights"
                )

            if strategy != "none":
                # Strategy descriptions
                strategy_descriptions = {
                    "class_weights": "Uses class weights in the model to penalize misclassification of minority classes. No data is discarded or duplicated.",
                    "undersample": "Randomly reduces majority class samples to match minority class size. Fast and effective for large datasets.",
                    "oversample": "Randomly duplicates minority class samples to match majority class size. Keeps all original data.",
                    "combined": "Combines undersampling (reduces majority to 3:1 ratio) with class weights for optimal balance.",
                }

                imbalance_md = f"""## 5.1 Handling Class Imbalance
**⚠️ Severe class imbalance detected!**

When one class has far more samples than another, models tend to ignore the minority class and just predict the majority class. This gives high accuracy but poor real-world performance.

**Selected Strategy**: {strategy.replace("_", " ").title()}
- {strategy_descriptions.get(strategy, "Custom strategy")}
"""
                nb.cells.append(nbformat.v4.new_markdown_cell(imbalance_md))

                # Generate code based on strategy
                if strategy == "undersample":
                    code7b = """# Check class distribution
print("Class distribution before resampling:")
class_counts = pd.Series(y_train).value_counts()
print(class_counts)
min_count = class_counts.min()
if min_count == 0:
    print("\\n⚠️  Warning: One or more classes have 0 samples in training set")
else:
    print(f"\\nImbalance ratio: {class_counts.max() / min_count:.1f}:1")

# Perform random undersampling
import numpy as np

# Convert to numpy arrays if needed
if not isinstance(X_train, np.ndarray):
    X_train = X_train.values
if not isinstance(y_train, np.ndarray):
    y_train = y_train.values

min_class_size = class_counts.min()
X_train_balanced = []
y_train_balanced = []

for class_label in class_counts.index:
    class_indices = np.where(y_train == class_label)[0]
    
    if len(class_indices) > min_class_size:
        sampled_indices = np.random.choice(class_indices, size=min_class_size, replace=False)
    else:
        sampled_indices = class_indices
    
    X_train_balanced.append(X_train[sampled_indices])
    y_train_balanced.extend(y_train[sampled_indices])

X_train = np.vstack(X_train_balanced)
y_train = np.array(y_train_balanced)

shuffle_indices = np.random.permutation(len(y_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print("\\n✓ Undersampling complete!")
print(f"New training set shape: {X_train.shape}")
print("\\nClass distribution after resampling:")
print(pd.Series(y_train).value_counts())
"""
                elif strategy == "oversample":
                    code7b = """# Check class distribution
print("Class distribution before resampling:")
class_counts = pd.Series(y_train).value_counts()
print(class_counts)

# Perform random oversampling
import numpy as np

# Convert to numpy arrays if needed
if not isinstance(X_train, np.ndarray):
    X_train = X_train.values
if not isinstance(y_train, np.ndarray):
    y_train = y_train.values

max_class_size = class_counts.max()
X_train_balanced = []
y_train_balanced = []

for class_label in class_counts.index:
    class_indices = np.where(y_train == class_label)[0]
    
    if len(class_indices) < max_class_size:
        # Oversample minority class
        sampled_indices = np.random.choice(class_indices, size=max_class_size, replace=True)
    else:
        sampled_indices = class_indices
    
    X_train_balanced.append(X_train[sampled_indices])
    y_train_balanced.extend(y_train[sampled_indices])

X_train = np.vstack(X_train_balanced)
y_train = np.array(y_train_balanced)

shuffle_indices = np.random.permutation(len(y_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print("\\n✓ Oversampling complete!")
print(f"New training set shape: {X_train.shape}")
print("\\nClass distribution after resampling:")
print(pd.Series(y_train).value_counts())
"""
                elif strategy == "combined":
                    code7b = """# Check class distribution
print("Class distribution before resampling:")
class_counts = pd.Series(y_train).value_counts()
print(class_counts)

# Combined approach: Undersample to 3:1 ratio
import numpy as np

# Convert to numpy arrays if needed
if not isinstance(X_train, np.ndarray):
    X_train = X_train.values
if not isinstance(y_train, np.ndarray):
    y_train = y_train.values

min_class_size = class_counts.min()
target_majority_size = min_class_size * 3  # 3:1 ratio

X_train_balanced = []
y_train_balanced = []

for class_label in class_counts.index:
    class_indices = np.where(y_train == class_label)[0]
    
    if len(class_indices) > target_majority_size:
        sampled_indices = np.random.choice(class_indices, size=target_majority_size, replace=False)
    else:
        sampled_indices = class_indices
    
    X_train_balanced.append(X_train[sampled_indices])
    y_train_balanced.extend(y_train[sampled_indices])

X_train = np.vstack(X_train_balanced)
y_train = np.array(y_train_balanced)

shuffle_indices = np.random.permutation(len(y_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

print("\\n✓ Combined resampling complete (3:1 ratio + class weights will be used in model)!")
print(f"New training set shape: {X_train.shape}")
print("\\nClass distribution after resampling:")
print(pd.Series(y_train).value_counts())
"""
                else:  # class_weights
                    code7b = """# Check class distribution
print("Class distribution:")
class_counts = pd.Series(y_train).value_counts()
print(class_counts)
min_count = class_counts.min()
if min_count == 0:
    print("\\n⚠️  Warning: One or more classes have 0 samples in training set")
else:
    print(f"\\nImbalance ratio: {class_counts.max() / min_count:.1f}:1")
print("\\n✓ Using class weights in model (no resampling needed)")
"""

                nb.cells.append(nbformat.v4.new_code_cell(code7b))

    # Cross-Validation Section
    cv_md = """## 6. K-Fold Cross-Validation
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
    eval_md = """## 7. Final Model Training and Evaluation
Train the final model on the training set and evaluate on the test set."""
    nb.cells.append(nbformat.v4.new_markdown_cell(eval_md))

    # Model fitting and test set evaluation cell
    if is_classifier:
        # Get target column information for context
        target_col = session_manager.get_field(session_id, "target_column")
        df = session_manager.get_field(session_id, "dataframe")
        target_dist = df[target_col].value_counts(normalize=True).sort_index()
        n_classes = len(target_dist)

        # Determine if binary or multiclass
        is_binary = n_classes == 2

        if is_binary:
            # Binary classification - unified output
            code9 = f"""# Train the model
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

model = {model_ctor}
model.fit(X_train, y_train)

# Evaluate on TRAINING set first (to detect overfitting)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on TEST set
y_pred = model.predict(X_test)

# Calculate comprehensive metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Calculate class distribution for context
unique, counts = np.unique(y_test, return_counts=True)
class_dist = dict(zip(unique, counts / len(y_test) * 100))

# Display results with smart context
print("\\n📊 MODEL PERFORMANCE SUMMARY")
print("─" * 60)
print(f"✓ Training Accuracy: {{train_accuracy:.2%}}")
print(f"✓ Test Accuracy:     {{accuracy:.2%}}")

# Overfitting detection with smart messaging
accuracy_diff = abs(train_accuracy - accuracy)
if accuracy_diff > 0.10:
    print(f"⚠️  {{accuracy_diff:.1%}} gap - significant overfitting detected!")
elif accuracy_diff > 0.05:
    print(f"⚠️  {{accuracy_diff:.1%}} gap - monitor for overfitting")
else:
    print(f"✓ {{accuracy_diff:.1%}} gap - good generalization")

print(f"\\nCore Metrics:")
print(f"  ✓ Accuracy:  {{accuracy:.2%}}")
print(f"  ✓ F1-Score:  {{f1:.2%}}")
print(f"  ✓ Precision: {{precision:.2%}}")
print(f"  ✓ Recall:    {{recall:.2%}}")

# Smart dataset context
min_class_pct = min(class_dist.values())
max_class_pct = max(class_dist.values())
imbalance_ratio = max_class_pct / min_class_pct

print(f"\\nℹ️  Dataset Info: ", end="")
if imbalance_ratio > 3:
    print(f"Severely imbalanced ({{max_class_pct:.2f}}%/{{min_class_pct:.2f}}%)")
    print("💡 Tip: Focus on F1-Score, Precision, and Recall rather than accuracy alone")
elif imbalance_ratio > 1.5:
    print(f"Moderately imbalanced ({{max_class_pct:.2f}}%/{{min_class_pct:.2f}}%)")
    print("💡 Tip: F1-Score is more reliable than accuracy for imbalanced data")
else:
    print(f"Well balanced ({{max_class_pct:.2f}}%/{{min_class_pct:.2f}}%)")

print("\\n🔍 CONFUSION MATRIX")
print("─" * 60)
print("                 Predicted")
print("               Negative  Positive")
print(f"Actual Negative  {{cm[0,0]:>6}}    {{cm[0,1]:>6}}")
print(f"       Positive  {{cm[1,0]:>6}}    {{cm[1,1]:>6}}")

# Per-class metrics with smart insights
print("\\n📈 PER-CLASS BREAKDOWN")
print("─" * 60)
class_names = ['Negative', 'Positive']
class_f1_scores = []
# Calculate per-class metrics correctly using binary average for each class
for i, class_label in enumerate(sorted(set(y_test))):
    # Create binary mask: this class vs all others
    y_test_binary = (y_test == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)
    
    class_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    class_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    class_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    class_f1_scores.append((class_names[i], class_f1))
    print(f"{{class_names[i]:>8}}: F1={{class_f1:.2%}}, Precision={{class_precision:.2%}}, Recall={{class_recall:.2%}}")

# Smart insights based on performance difference
if len(class_f1_scores) == 2:
    diff = abs(class_f1_scores[0][1] - class_f1_scores[1][1])
    if diff > 0.15:
        worse_class = min(class_f1_scores, key=lambda x: x[1])[0]
        print(f"\\n⚠️  Model performs significantly worse on {{worse_class}} class")
        if imbalance_ratio > 2:
            print("   This is common with imbalanced datasets - consider collecting more minority class samples")

# Save trained model
with open(f'artifacts/model_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\\n✅ Model saved successfully!")
"""
        else:
            # Multiclass classification - unified output
            code9 = f"""# Train the model
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

model = {model_ctor}
model.fit(X_train, y_train)

# Evaluate on TRAINING set first (to detect overfitting)
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Evaluate on TEST set
y_pred = model.predict(X_test)

# Calculate comprehensive metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Calculate class distribution for context
unique, counts = np.unique(y_test, return_counts=True)
class_dist = dict(zip(unique, counts / len(y_test) * 100))

# Display results with smart context
print("\\n📊 MODEL PERFORMANCE SUMMARY")
print("─" * 60)
print(f"✓ Training Accuracy: {{train_accuracy:.2%}}")
print(f"✓ Test Accuracy:     {{accuracy:.2%}}")

# Overfitting detection with smart messaging
accuracy_diff = abs(train_accuracy - accuracy)
if accuracy_diff > 0.10:
    print(f"⚠️  {{accuracy_diff:.1%}} gap - significant overfitting detected!")
elif accuracy_diff > 0.05:
    print(f"⚠️  {{accuracy_diff:.1%}} gap - monitor for overfitting")
else:
    print(f"✓ {{accuracy_diff:.1%}} gap - good generalization")

print(f"\\nCore Metrics (Weighted Average):")
print(f"  ✓ Accuracy:  {{accuracy:.2%}}")
print(f"  ✓ F1-Score:  {{f1:.2%}}")
print(f"  ✓ Precision: {{precision:.2%}}")
print(f"  ✓ Recall:    {{recall:.2%}}")

# Smart dataset context
min_class_pct = min(class_dist.values())
max_class_pct = max(class_dist.values())
imbalance_ratio = max_class_pct / min_class_pct

print(f"\\nℹ️  Dataset Info: {{len(class_dist)}} classes", end="")
if imbalance_ratio > 2.5:
    print(f" - Severely imbalanced (largest: {{max_class_pct:.2f}}%, smallest: {{min_class_pct:.2f}}%)")
    print("💡 Tip: Check per-class metrics below - minority classes may have lower performance")
elif imbalance_ratio > 1.5:
    print(f" - Moderately imbalanced (largest: {{max_class_pct:.2f}}%, smallest: {{min_class_pct:.2f}}%)")
else:
    print(f" - Reasonably balanced")

print("\\n🔍 CONFUSION MATRIX")
print("─" * 60)
print("                 Predicted")
print("               ", end="")
for i in range(cm.shape[1]):
    print(f"Class {{i}}  ", end="")
print()
for i in range(cm.shape[0]):
    print(f"Actual Class {{i}} ", end="")
    for j in range(cm.shape[1]):
        print(f"{{cm[i,j]:>7}}  ", end="")
    print()

# Per-class metrics with smart insights
print("\\n📈 PER-CLASS BREAKDOWN")
print("─" * 60)
class_f1_scores = []
# Calculate per-class metrics correctly using binary classification for each class
for class_label in sorted(set(y_test)):
    # Create binary mask: this class vs all others
    y_test_binary = (y_test == class_label).astype(int)
    y_pred_binary = (y_pred == class_label).astype(int)
    
    class_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    class_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    class_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    class_f1_scores.append((class_label, class_f1))
    print(f"Class {{class_label}}: F1={{class_f1:.2%}}, Precision={{class_precision:.2%}}, Recall={{class_recall:.2%}}")

# Smart insights based on performance variance
if len(class_f1_scores) > 2:
    f1_values = [score for _, score in class_f1_scores]
    f1_std = np.std(f1_values)
    if f1_std > 0.10:
        worst_class = min(class_f1_scores, key=lambda x: x[1])[0]
        best_class = max(class_f1_scores, key=lambda x: x[1])[0]
        print(f"\\n⚠️  High variance in per-class performance")
        print(f"   Best: Class {{best_class}}, Worst: Class {{worst_class}}")
        if imbalance_ratio > 2:
            print("   Consider collecting more samples for underperforming classes")

# Save trained model
with open(f'artifacts/model_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\\n✅ Model saved successfully!")
"""
    else:
        # Regression - unified output with smart context
        code9 = f"""# Train the model
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

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

# Calculate metrics for interpretation
r2_diff = abs(train_r2 - test_r2)
rmse_increase = ((test_rmse - train_rmse) / train_rmse * 100) if train_rmse > 0 else 0
target_std = np.std(y_test)
rmse_ratio = test_rmse / target_std if target_std > 0 else 0

# Display results with smart context
print("\\n📊 REGRESSION PERFORMANCE SUMMARY")
print("─" * 60)
print("Training Set:")
print(f"  ✓ R² Score: {{train_r2:.4f}}")
print(f"  ✓ RMSE:     {{train_rmse:.4f}}")
print(f"  ✓ MAE:      {{train_mae:.4f}}")

print("\\nTest Set:")
print(f"  ✓ R² Score: {{test_r2:.4f}}")
print(f"  ✓ RMSE:     {{test_rmse:.4f}}")
print(f"  ✓ MAE:      {{test_mae:.4f}}")

# Smart overfitting detection for regression
print()
if r2_diff > 0.15:
    print(f"⚠️  R² dropped {{r2_diff:.2f}} from train to test - significant overfitting!")
    print("   Consider: more data, feature selection, or regularization")
elif r2_diff > 0.08:
    print(f"⚠️  R² dropped {{r2_diff:.2f}} from train to test - monitor for overfitting")
else:
    print(f"✓ R² dropped only {{r2_diff:.2f}} - good generalization")

if rmse_increase > 30:
    print(f"⚠️  RMSE increased {{rmse_increase:.1f}}% on test set")

# Model quality interpretation
print("\\n💡 Model Interpretation:")
if test_r2 < 0:
    print("  ⚠️  Negative R² - model performs worse than predicting the mean")
    print("     Consider: feature engineering, different model, or check data quality")
elif test_r2 < 0.3:
    print(f"  ⚠️  Low R² ({{test_r2:.2f}}) - model explains little variance")
    print("     Consider: adding more relevant features or trying different models")
elif test_r2 < 0.7:
    print(f"  ✓ Moderate R² ({{test_r2:.2f}}) - model captures some patterns")
    print("     Room for improvement with feature engineering")
else:
    print(f"  ✓ Good R² ({{test_r2:.2f}}) - model explains most variance")

# RMSE interpretation
print(f"\\n  • RMSE of {{test_rmse:.4f}} means predictions are off by ±{{test_rmse:.4f}} units on average")
if rmse_ratio < 0.3:
    print(f"  • RMSE is {{rmse_ratio:.1%}} of target std dev - excellent precision")
elif rmse_ratio < 0.5:
    print(f"  • RMSE is {{rmse_ratio:.1%}} of target std dev - good precision")
elif rmse_ratio < 0.8:
    print(f"  • RMSE is {{rmse_ratio:.1%}} of target std dev - moderate precision")
else:
    print(f"  • RMSE is {{rmse_ratio:.1%}} of target std dev - consider improving model")

# MAE interpretation
print(f"  • MAE of {{test_mae:.4f}} means typical prediction error is {{test_mae:.4f}} units")
if test_mae < test_rmse * 0.8:
    print("  • MAE < RMSE indicates some large outlier errors exist")

# Save trained model
with open(f'artifacts/model_{{SESSION_ID}}.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\\n✅ Model saved successfully!")
"""
    nb.cells.append(nbformat.v4.new_code_cell(code9))

    # Summary and Usage Instructions
    summary_md = f"""## 8. Summary and Next Steps

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
    session_manager.set_field(
        session_id,
        "artifacts",
        {
            "model": os.path.join(ARTIFACTS_FOLDER, f"model_{session_id}.pkl"),
            "scaler": os.path.join(ARTIFACTS_FOLDER, f"scaler_{session_id}.pkl"),
            "notebook": f"{session_id}.ipynb",
        },
    )

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

            # Get cell title from session
            cell_titles = session_manager.get_field(session_id, "cell_titles", [])
            cell_title = (
                cell_titles[code_cell_counter[0] - 1]
                if code_cell_counter[0] <= len(cell_titles)
                else f"Cell {code_cell_counter[0]}"
            )

            # Store result for LLM summary
            all_cell_results.append(
                {"cell_number": code_cell_counter[0], "result": cell_result}
            )

            # Put result in queue for streaming (now includes title)
            result_queue.put(
                {
                    "cell": code_cell_counter[0],
                    "result": cell_result,
                    "title": cell_title,
                }
            )

        def execute_notebook():
            """Execute notebook in a separate thread"""
            try:
                nb_to_run = copy.deepcopy(nb)
                # Execute from project root so relative paths work correctly
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
                session_manager.set_field(session_id, "cell_results", all_cell_results)

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
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "artifacts") is None:
        raise HTTPException(
            status_code=404, detail="No model artifacts found for this session"
        )

    model_file = session_manager.get_field(session_id, "artifacts")["model"]
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        model_file,
        media_type="application/octet-stream",
        filename=f"model_{session_id}.pkl",
    )


@app.get("/download/scaler/{session_id}")
async def download_scaler(session_id: str):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "artifacts") is None:
        raise HTTPException(
            status_code=404, detail="No scaler artifacts found for this session"
        )

    scaler_file = session_manager.get_field(session_id, "artifacts")["scaler"]
    if not os.path.exists(scaler_file):
        raise HTTPException(status_code=404, detail="Scaler file not found")

    return FileResponse(
        scaler_file,
        media_type="application/octet-stream",
        filename=f"scaler_{session_id}.pkl",
    )


@app.get("/download/notebook/{session_id}")
async def download_notebook(session_id: str):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "artifacts") is None:
        raise HTTPException(
            status_code=404, detail="No notebook artifacts found for this session"
        )

    notebook_file = os.path.join(
        NOTEBOOKS_FOLDER, session_manager.get_field(session_id, "artifacts")["notebook"]
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
        cell_results = session_manager.get_field(session_id, "cell_results")
        target_col = session_manager.get_field(session_id, "target_column", "unknown")
        model_name = session_manager.get_field(session_id, "model", "unknown")
        dataset_filename = session_manager.get_field(session_id, "dataset", "")
        dataset_name = dataset_filename.replace(session_id, "").replace(".csv", "")

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
        session_manager.set_field(
            session_id,
            "ai_report",
            {
                "status": "completed",
                "report": report,
                "generated_at": pd.Timestamp.now().isoformat(),
            },
        )

    except Exception as e:
        import traceback

        # Store error in session
        session_manager.set_field(
            session_id,
            "ai_report",
            {
                "status": "failed",
                "error": f"{str(e)}\n{traceback.format_exc()}",
            },
        )


@app.api_route("/generate/report/{session_id}", methods=["GET", "POST"])
async def generate_report(session_id: str, background_tasks: BackgroundTasks):
    """Start AI-powered report generation in background"""
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    if session_manager.get_field(session_id, "cell_results") is None:
        raise HTTPException(
            status_code=404,
            detail="No execution results found. Please run the notebook first.",
        )

    # Check if report is already being generated or completed
    if session_manager.get_field(session_id, "ai_report") is not None:
        existing_status = session_manager.get_field(session_id, "ai_report").get(
            "status"
        )
        if existing_status == "completed":
            return session_manager.get_field(session_id, "ai_report")
        elif existing_status == "generating":
            return {
                "session_id": session_id,
                "status": "generating",
                "message": "Report generation already in progress. Use /report/status/{session_id} to check progress.",
            }

    # Mark report as generating
    session_manager.set_field(
        session_id,
        "ai_report",
        {
            "status": "generating",
            "message": "Report generation in progress...",
        },
    )

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
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    if session_manager.get_field(session_id, "ai_report") is None:
        return {"status": "not_started", "message": "Report not requested yet"}

    return session_manager.get_field(session_id, "ai_report")


@app.get("/report/stream/{session_id}")
async def stream_report_status(session_id: str):
    """Stream report generation status via Server-Sent Events"""
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    async def event_generator():
        """Generate SSE events for report status"""
        while True:
            if session_manager.get_field(session_id, "ai_report") is None:
                yield f"data: {json.dumps({'status': 'not_started'})}\n\n"
                await asyncio.sleep(1)
                continue

            report_data = session_manager.get_field(session_id, "ai_report")
            yield f"data: {json.dumps(report_data)}\n\n"

            # Stop streaming if completed or failed
            if report_data.get("status") in ["completed", "failed"]:
                break

            await asyncio.sleep(2)  # Poll every 2 seconds

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def validate_prediction_input(input_data: dict, session_id: str) -> Dict[str, Any]:
    """Validate prediction input against training schema."""

    # Load the actual input columns that were used for training (after exclusions)
    input_cols_file = os.path.join(ARTIFACTS_FOLDER, f"input_columns_{session_id}.pkl")
    if not os.path.exists(input_cols_file):
        raise HTTPException(
            status_code=404,
            detail="Model not trained yet. Please train the model first.",
        )

    with open(input_cols_file, "rb") as f:
        required_cols = pickle.load(f)

    # Load training data for validation ranges (only needed columns)
    dataset_filename = session_manager.get_field(session_id, "dataset")
    if not dataset_filename:
        raise HTTPException(status_code=400, detail="Training dataset not found")

    df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_filename))
    target_col = session_manager.get_field(session_id, "target_column")
    X_train = df.drop(columns=[target_col])

    # Filter X_train to only include columns that were actually used for training
    X_train = X_train[required_cols]

    validation_errors = []

    # Check for missing required columns
    provided_cols = set(input_data.keys())
    missing_cols = set(required_cols) - provided_cols

    if missing_cols:
        validation_errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for extra columns not in training
    extra_cols = provided_cols - set(required_cols)
    if extra_cols:
        validation_errors.append(
            f"Unknown columns (not in training data): {', '.join(extra_cols)}"
        )

    # Validate data types and ranges for each column
    for col in X_train.columns:
        if col not in input_data:
            continue

        value = input_data[col]

        # Check numeric columns
        if pd.api.types.is_numeric_dtype(X_train[col]):
            try:
                float_val = float(value)

                # Validate using min/max from training data with reasonable buffer
                # This allows for natural variation and outliers that models can handle
                col_min = X_train[col].min()
                col_max = X_train[col].max()

                # Allow 50% extrapolation beyond training range
                # This handles legitimate outliers while catching obvious errors
                buffer_factor = 0.5
                range_span = col_max - col_min

                # Handle edge case where all training values are the same
                if range_span == 0:
                    range_span = abs(col_min) if col_min != 0 else 1.0

                min_val = col_min - (buffer_factor * range_span)
                max_val = col_max + (buffer_factor * range_span)

                # For columns that should be non-negative (common in real data)
                # Don't allow negative values if training data was all non-negative
                if col_min >= 0:
                    min_val = max(0, min_val)

                if float_val < min_val or float_val > max_val:
                    validation_errors.append(
                        f"Column '{col}': value {float_val} is outside reasonable range "
                        f"[{min_val:.2f}, {max_val:.2f}]. Training data ranged from {col_min:.2f} to {col_max:.2f}"
                    )
            except (ValueError, TypeError):
                validation_errors.append(
                    f"Column '{col}': expected numeric value, got '{value}'"
                )

        # For categorical columns, just ensure it's a string (no strict validation)
        # The frontend dropdown already constrains users to valid values
        # For API calls, the prediction logic handles unknown values gracefully
        elif pd.api.types.is_object_dtype(X_train[col]):
            if not isinstance(value, (str, int, float)):
                validation_errors.append(
                    f"Column '{col}': expected string or numeric value, got {type(value).__name__}"
                )

    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation failed", "errors": validation_errors},
        )

    return input_data


@app.post("/predict/{session_id}")
@limiter.limit("200/minute")
async def predict(request: Request, session_id: str, input_data: dict):
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if session_manager.get_field(session_id, "artifacts") is None:
        raise HTTPException(status_code=404, detail="Model not trained yet")

    # Validate input data
    try:
        input_data = validate_prediction_input(input_data, session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    model_file = session_manager.get_field(session_id, "artifacts")["model"]
    scaler_file = session_manager.get_field(session_id, "artifacts")["scaler"]

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

        # Load the actual input columns that were used for training
        input_cols_file = os.path.join(
            ARTIFACTS_FOLDER, f"input_columns_{session_id}.pkl"
        )
        with open(input_cols_file, "rb") as f:
            training_columns = pickle.load(f)

        # Get original dataset to understand feature types (only for columns used in training)
        dataset_filename = session_manager.get_field(session_id, "dataset")
        df = pd.read_csv(os.path.join(DATA_FOLDER, dataset_filename))
        target_col = session_manager.get_field(session_id, "target_column")
        X_original = df.drop(columns=[target_col])

        # Filter to only columns that were actually used for training
        X_original = X_original[training_columns]

        # Convert input_data values to correct types (handle string inputs from frontend)
        typed_input_data = {}
        for col in training_columns:
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

        # Step 1: Handle missing values AND create missing indicators (MUST match training)
        for col in training_columns:
            col_exists = col in input_df.columns
            col_is_null = col_exists and pd.isna(input_df[col].iloc[0])

            if not col_exists or col_is_null:
                if pd.api.types.is_numeric_dtype(X_original[col]):
                    unique_vals = X_original[col].dropna().unique()

                    # Binary numeric (0/1) - Create missing indicator (match training logic exactly)
                    if len(unique_vals) <= 2 and set(unique_vals).issubset(
                        {0, 1, 0.0, 1.0}
                    ):
                        missing_indicator_col = f"{col}_missing"
                        input_df[missing_indicator_col] = 1  # Mark as missing
                        input_df[col] = 0  # Fill with 0
                    else:
                        # Continuous numeric - Use median
                        input_df[col] = X_original[col].median()
                else:
                    # Categorical - check missing percentage in training
                    missing_pct = (
                        X_original[col].isnull().sum() / len(X_original)
                    ) * 100
                    if missing_pct < 5:
                        # Low missing - use mode
                        mode_val = X_original[col].mode()
                        input_df[col] = mode_val[0] if not mode_val.empty else "Unknown"
                    else:
                        # High missing - use "Missing" category
                        input_df[col] = "Missing"

        # Step 2: Also check existing columns for binary numerics that have values
        for col in training_columns:
            if col in input_df.columns and not pd.isna(input_df[col].iloc[0]):
                if pd.api.types.is_numeric_dtype(X_original[col]):
                    unique_vals = X_original[col].dropna().unique()
                    # Binary numeric with value present - still need to create indicator (set to 0)
                    if len(unique_vals) <= 2 and set(unique_vals).issubset(
                        {0, 1, 0.0, 1.0}
                    ):
                        missing_indicator_col = f"{col}_missing"
                        if missing_indicator_col not in input_df.columns:
                            input_df[missing_indicator_col] = 0  # Not missing

        # Don't reorder columns here - the final alignment step (after all preprocessing)
        # will handle column ordering using the features_{session_id}.pkl file
        # This ensures _missing indicators and all other generated columns are properly aligned

        # Apply same preprocessing as training
        # 1. Identify text and categorical columns (MUST match training logic exactly)
        text_cols = []
        cat_cols = []
        date_cols = []

        for col in input_df.select_dtypes(include=["object", "category"]).columns:
            # Date detection (match training: sample multiple rows, require >50% valid)
            def is_valid_date(val_str):
                try:
                    pd.to_datetime(val_str)
                    return True
                except Exception:
                    return False

            # Sample up to 10 rows from training data for date detection
            sample_data = X_original[col].head(min(10, len(X_original)))
            valid_dates = sum(1 for val in sample_data if is_valid_date(str(val)))
            if len(sample_data) > 0 and valid_dates / len(sample_data) > 0.5:
                date_cols.append(col)
                continue

            # Text vs Categorical detection (match training: check ratio AND string length)
            uniqueness_ratio = X_original[col].nunique() / len(X_original)
            sample_strings = X_original[col].dropna().head(100).astype(str)
            avg_str_len = (
                sample_strings.str.len().mean() if len(sample_strings) > 0 else 0
            )

            # Match training logic exactly: (ratio > 0.5 AND nunique > 50) OR avg_length > 50
            if (
                uniqueness_ratio > 0.5 and X_original[col].nunique() > 50
            ) or avg_str_len > 50:
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

        # 3. Encode categorical columns (MUST match training: split by cardinality)
        low_card_cats = []
        high_card_cats = []

        for col in cat_cols:
            n_unique = X_original[col].nunique()
            if n_unique <= 20:  # Low cardinality - one-hot encode
                low_card_cats.append(col)
            else:  # High cardinality - label encode
                high_card_cats.append(col)

        # One-hot encode low-cardinality categoricals
        if low_card_cats:
            input_df = pd.get_dummies(input_df, columns=low_card_cats, drop_first=True)

        # Label encode high-cardinality categoricals (load encoder from training)
        for col in high_card_cats:
            le_file = os.path.join(
                ARTIFACTS_FOLDER, f"label_encoder_{col}_{session_id}.pkl"
            )
            if os.path.exists(le_file):
                with open(le_file, "rb") as f:
                    le = pickle.load(f)
                # Handle unknown categories
                input_df[col] = input_df[col].astype(str)
                known_categories = set(le.classes_)
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in known_categories else le.classes_[0]
                )
                input_df[col] = le.transform(input_df[col])
            else:
                # Fallback: treat as numeric if encoder not found
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # 4. Combine all features
        if tfidf_features:
            input_df = pd.concat([input_df] + tfidf_features, axis=1)

        # 5. Align columns with training data FIRST (before scaling)
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

        # 6. Scale numeric features AFTER aligning
        if scaler:
            # After alignment, input_df has all expected features (missing ones added as 0)
            # Pass only the numeric columns to the scaler in the same order as training
            if hasattr(scaler, "feature_names_in_"):
                # Use scaler's feature names to ensure correct column order and selection
                input_df[scaler.feature_names_in_] = scaler.transform(
                    input_df[scaler.feature_names_in_]
                )

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

        # Clear all sessions from Redis
        session_count = session_manager.clear_all()

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

    if not session_manager.exists(session_id):
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

        target_column = session_manager.get_field(session_id, "target_column", "")

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

    if not session_manager.exists(session_id):
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
    all_session_ids = session_manager.list_all_sessions()

    for sid in all_session_ids:
        data = session_manager.get(sid)
        if data:
            session_info = {
                "session_id": sid,
                "dataset": data.get("dataset", "N/A"),
                "target_column": data.get("target_column", "N/A"),
                "model": data.get("model", "N/A"),
                "has_artifacts": "artifacts" in data,
            }
            session_list.append(session_info)

    return {"total_sessions": len(session_list), "sessions": session_list}


@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up all files and data for a specific session"""
    if not session_manager.exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        session_data = session_manager.get(session_id)
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
        session_manager.delete(session_id)

        return {
            "message": "Session cleaned up successfully",
            "files_deleted": files_deleted,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
