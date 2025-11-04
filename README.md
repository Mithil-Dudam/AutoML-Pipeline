# 🤖 AutoML Pipeline

An automated machine learning pipeline generator that creates complete, executable Jupyter notebooks from your dataset with just a few clicks. No coding required!

[![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)](https://github.com/Mithil-Dudam/AutoML-Pipeline)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

## 🌟 Features

### 🎯 Automated ML Pipeline

- **Smart Data Analysis**: Automatic EDA with dataset statistics, target distribution, and missing value detection
- **Intelligent Preprocessing**:
  - Automatic handling of missing values (mean/mode imputation)
  - TF-IDF vectorization for text columns
  - One-hot encoding for categorical features
  - StandardScaler for numeric features (when needed)
- **Model Selection**: Support for 15+ algorithms including:
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, KNN, Naive Bayes, Gradient Boosting, LightGBM, Extra Trees
  - **Regression**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVR, KNN
- **Cross-Validation**: 5-fold CV with stratification for classification
- **Model Evaluation**: Comprehensive metrics (accuracy, R², MSE, classification reports)

### 📊 Real-Time Execution

- **Live Streaming**: Watch your notebook execute cell-by-cell in real-time
- **Instant Feedback**: See results as they're generated using Server-Sent Events (SSE)
- **Error Handling**: Clear error messages for debugging

### 💾 Export & Reuse

- **Download Trained Models**: Get your `.pkl` model file ready for deployment
- **Download Scalers & Vectorizers**: All preprocessing artifacts included
- **Download Notebooks**: Fully executable `.ipynb` files for further customization
- **Reproducible**: All notebooks are self-contained and portable

### 🤖 AI-Powered Analysis Report

- **LLM-Generated Insights**: Get expert-level analysis of your ML pipeline results using Llama 3.2
- **Comprehensive Reports**: Automatically generated reports with:
  - Executive Summary
  - Data Quality Assessment
  - Model Performance Analysis
  - Key Findings & Insights
  - Actionable Recommendations
  - Deployment Readiness Assessment
- **Plain English Explanations**: Non-technical users can understand model performance
- **Local & Private**: Uses Ollama for privacy-preserving AI analysis

### 🔮 Prediction API

- **REST API**: Make predictions on new data using your trained model
- **Smart Input Validation**: Validates data types, ranges, and missing columns
- **Automatic Preprocessing**: Input data is transformed using saved scalers/vectorizers
- **Categorical Dropdowns**: Frontend automatically shows dropdowns for categorical features
- **JSON Interface**: Easy integration with any application

### 🧹 Session Management

- **Redis-Based Sessions**: Persistent session storage with 2-hour TTL
- **Multi-Session Support**: Handle multiple datasets simultaneously
- **Automatic Cleanup**: Background task removes expired sessions and orphaned files every 10 minutes
- **Session Tracking**: Monitor all active sessions with detailed stats

### 🔒 Production Features

- **Rate Limiting**: Configurable rate limiting (100 req/min default) to prevent API abuse
- **Smart File Validation**:
  - Automatic encoding detection (UTF-8, Latin-1, CP1252, ISO-8859-1)
  - Automatic delimiter detection (comma, semicolon, tab)
  - Better error messages for malformed files
- **Structured Logging**:
  - Request ID tracking for debugging
  - JSON-formatted logs with timestamps
  - Console output (captured by Docker)
- **Input Validation**: Comprehensive Pydantic-based validation for predictions
  - Type checking for all input fields
  - Range validation for numeric values (3σ from training mean)
  - Missing/extra column detection

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (Recommended)
  - Docker Desktop for Windows/Mac
  - Docker Engine + Docker Compose for Linux

**OR** for manual installation:

- Python 3.11+
- Node.js 18+
- npm or yarn
- **Redis 7+** (for session management)
- **Ollama** (for AI report generation)
  - Download from: [https://ollama.ai](https://ollama.ai)
  - Install the `llama3.2` model: `ollama pull llama3.2`

### Installation

#### Option 1: Docker (Recommended) 🐳

1. **Clone the repository**

```bash
git clone https://github.com/Mithil-Dudam/AutoML-Pipeline.git
cd AutoML-Pipeline
```

2. **Configure environment variables** (optional)

```bash
# Create .env file with your settings (see Configuration section below)
# Default settings work out of the box
```

3. **Start all services**

```bash
# Build and start all containers (Redis, Ollama, Backend, Frontend)
docker compose up --build

# Or run in detached mode
docker compose up -d
```

4. **Access the application**

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

5. **Stop the application**

```bash
docker compose down
```

#### Option 2: Manual Installation

1. **Clone the repository**

```bash
git clone https://github.com/Mithil-Dudam/AutoML-Pipeline.git
cd AutoML-Pipeline
```

2. **Install and start Redis**

```bash
# Windows (using Chocolatey)
choco install redis

# Mac
brew install redis
brew services start redis

# Linux (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis
```

3. **Set up Python backend**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

4. **Set up React frontend**

```bash
cd app_ui
npm install
cd ..
```

5. **Configure environment variables**

```bash
# Create .env file with your settings (see Configuration section below)
# For local development, use these values:
# REDIS_HOST=localhost
# REDIS_PORT=6379
```

6. **Start Ollama** (for AI report generation)

```bash
# Download and install from https://ollama.ai
# Then pull the model
ollama pull llama3.2

# Start Ollama service
ollama serve
```

7. **Start the backend server**

```bash
# From the root directory
uvicorn main:app --reload --port 8000
```

8. **Start the frontend development server**

```bash
# In a new terminal, from app_ui directory
cd app_ui
npm run dev
```

9. **Open your browser**

```
http://localhost:5173
```

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the root directory if you need to customize Redis settings:

```env
# Redis configuration
REDIS_HOST=redis                # Use "redis" for Docker, "localhost" for local
REDIS_PORT=6379                 # Default Redis port
```

**Note:** The application works out-of-the-box with default settings. You only need a `.env` file if you're running Redis on a different host/port.

## 📖 Usage Guide

### Step 1: Upload Dataset

- Click "Choose File" and select your CSV dataset
- File is automatically uploaded and assigned a unique session ID

### Step 2: Configure

- **Select Target Column**: Choose which column you want to predict
  - Automatic detection of useless columns (IDs, emails, URLs, phone numbers, names, hashes)
  - Smart warnings for class imbalance, binary transformation opportunities, etc.
  - Recommended strategies for handling imbalanced datasets
- **Review Columns**: Review and exclude columns before training
  - See data types, unique values, and sample data
  - Auto-excluded columns are highlighted with reasons
- **Select Model**: Pick from 19+ classification or regression algorithms
- **Handle Class Imbalance** (if detected):
  - Random Undersampling
  - Random Oversampling
  - SMOTE (Synthetic Minority Over-sampling)
  - Combined (SMOTE + Undersampling)
  - No resampling

### Step 3: Generate & Execute

- Click "Generate Notebook"
- Watch in real-time as the notebook is created and executed
- See results streaming live for each cell

### Step 4: Get AI Analysis Report

- Click "Generate AI Report" button
- LLM (Llama 3.2) analyzes your results in real-time
- Get expert insights on:
  - Data quality
  - Model performance
  - Overfitting/underfitting detection
  - Actionable recommendations

### Step 5: Download & Use

- **Download Model**: Get your trained model (`.pkl`)
- **Download Scaler**: Get preprocessing artifacts
- **Download Notebook**: Get the full Jupyter notebook

### Step 6: Make Predictions (Optional)

- Use the test section to make predictions on new data
- **Categorical columns** show as dropdowns with valid values
- **Numeric columns** show as text inputs with validation
- Smart validation checks:
  - Missing/extra columns
  - Data type validation
  - Range validation (values within 3σ of training data)
- Get instant predictions from your trained model

## 🛠️ API Endpoints

### System Endpoints

- `GET /health` - Health check endpoint

### Dataset Management

- `POST /dataset` - Upload a CSV dataset (with smart encoding/delimiter detection)
- `POST /target-column` - Set target column for prediction
- `POST /model` - Select ML model
- `POST /imbalance-strategy` - Set class imbalance handling strategy
- `GET /column-info/{session_id}` - Get column information for review
- `POST /exclude-columns` - Exclude specific columns from training

### Notebook Operations

- `GET /generate/notebook` - Generate and execute notebook (SSE stream)
- `GET /download/notebook/{session_id}` - Download generated notebook

### Model Operations

- `GET /download/model/{session_id}` - Download trained model
- `GET /download/scaler/{session_id}` - Download scaler/preprocessors
- `POST /predict/{session_id}` - Make predictions with trained model (with validation)
- `GET /input-columns/{session_id}` - Get expected input columns for predictions
- `GET /categorical-values/{session_id}` - Get valid categorical values for dropdowns

### AI Analysis

- `POST /generate/report/{session_id}` - Start AI-powered analysis report generation
- `GET /report/status/{session_id}` - Check report generation status
- `GET /report/stream/{session_id}` - Stream report generation progress (SSE)

### Session Management

- `GET /sessions` - List all active sessions with details
- `DELETE /cleanup/{session_id}` - Clean up specific session
- `DELETE /cleanup/all` - Clean up all sessions

## 🔧 Technical Stack

### Backend

- **FastAPI**: Modern, fast web framework for building APIs
- **Redis**: In-memory database for session management
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **NBFormat**: Jupyter notebook format handling
- **NBClient**: Jupyter notebook execution
- **LangChain + Ollama**: LLM integration for AI-powered analysis reports
- **Pydantic**: Data validation and settings management
- **SlowAPI**: Rate limiting middleware
- **Chardet**: Character encoding detection

### Frontend

- **React 18**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing
- **Axios**: HTTP client for API requests

### Infrastructure

- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Frontend web server (in production container)

## 📊 Supported Algorithms

### Classification Models

| Algorithm           | Key Features              | Best For                           |
| ------------------- | ------------------------- | ---------------------------------- |
| Logistic Regression | Linear, interpretable     | Binary/multiclass classification   |
| Decision Tree       | Non-linear, interpretable | Complex patterns                   |
| Random Forest       | Ensemble, robust          | High accuracy, feature importance  |
| SVM                 | Kernel methods            | Small to medium datasets           |
| XGBoost             | Gradient boosting         | Competition-winning performance    |
| K-Nearest Neighbors | Instance-based            | Non-parametric problems            |
| Naive Bayes         | Probabilistic             | Text classification, fast training |
| Gradient Boosting   | Ensemble                  | High accuracy                      |
| LightGBM            | Fast gradient boosting    | Large datasets                     |
| Extra Trees         | Ensemble                  | Reduced overfitting                |

### Regression Models

| Algorithm           | Key Features                | Best For                  |
| ------------------- | --------------------------- | ------------------------- |
| Linear Regression   | Simple, interpretable       | Linear relationships      |
| Ridge Regression    | L2 regularization           | Multicollinearity         |
| Lasso Regression    | L1 regularization           | Feature selection         |
| Decision Tree       | Non-linear                  | Complex patterns          |
| Random Forest       | Ensemble                    | High accuracy             |
| Gradient Boosting   | Sequential ensemble         | Excellent performance     |
| XGBoost             | Optimized gradient boosting | Competition-level results |
| SVR                 | Kernel methods              | Non-linear relationships  |
| K-Nearest Neighbors | Instance-based              | Smooth predictions        |

## 🎓 Example Notebook Structure

Each generated notebook contains:

1. **Configuration**: Dataset paths and settings
2. **Installation**: Required packages
3. **Imports**: All necessary libraries
4. **EDA**:
   - Dataset preview (head)
   - Dataset info
   - Summary statistics
   - Target distribution (classification) or stats (regression)
5. **Data Cleaning**: Missing value detection and imputation
6. **Feature Engineering**:
   - TF-IDF for text
   - One-hot encoding for categoricals
   - Scaling for numerics
7. **Train-Test Split**: 80/20 split with stratification
8. **Cross-Validation**: 5-fold evaluation
9. **Model Training**: Final model on full training set
10. **Evaluation**: Metrics and performance reports
11. **Model Saving**: Export for deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Mithil Dudam**

- GitHub: [@Mithil-Dudam](https://github.com/Mithil-Dudam)

## 📧 Support

If you have any questions or run into issues, please open an issue on GitHub.

---

⭐ **Star this repository if you find it helpful!**
