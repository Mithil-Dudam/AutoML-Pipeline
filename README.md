# 🤖 AutoML Pipeline

An automated machine learning pipeline generator that creates complete, executable Jupyter notebooks from your dataset with just a few clicks. No coding required!

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

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
- **Automatic Preprocessing**: Input data is transformed using saved scalers/vectorizers
- **JSON Interface**: Easy integration with any application

### 🧹 Session Management

- **Multi-Session Support**: Handle multiple datasets simultaneously
- **Automatic Cleanup**: Remove old sessions and artifacts
- **Session Tracking**: Monitor all active sessions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- **Ollama** (for AI report generation)
  - Download from: [https://ollama.ai](https://ollama.ai)
  - Install the `llama3.2` model: `ollama pull llama3.2`

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Mithil-Dudam/AutoML-Pipeline.git
cd AutoML-Pipeline
```

2. **Set up Python backend**

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

3. **Set up React frontend**

```bash
cd app_ui
npm install
```

### Running the Application

1. **Start Ollama** (for AI report generation)

```bash
# Make sure Ollama is running in the background
ollama serve
```

2. **Start the backend server**

```bash
# From the root directory
uvicorn main:app --reload --port 8000
```

3. **Start the frontend development server**

```bash
# In a new terminal, from app_ui directory
cd app_ui
npm run dev
```

4. **Open your browser**

```
http://localhost:5173
```

## 📖 Usage Guide

### Step 1: Upload Dataset

- Click "Choose File" and select your CSV dataset
- File is automatically uploaded and assigned a unique session ID

### Step 2: Configure

- **Select Target Column**: Choose which column you want to predict
- **Select Model**: Pick from 15+ classification or regression algorithms

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
- Enter feature values in JSON format
- Get instant predictions from your trained model

## 🏗️ Project Structure

```
AutoML-Pipeline/
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── dataset/               # Uploaded datasets (gitignored)
├── notebooks/             # Generated notebooks (gitignored)
├── artifacts/             # Models, scalers (gitignored)
└── app_ui/                # React frontend
    ├── src/
    │   ├── pages/
    │   │   ├── Home.tsx        # Upload & configuration page
    │   │   ├── Results.tsx     # Notebook execution & results
    │   │   └── PageNotFound.tsx
    │   ├── Api.tsx             # API client
    │   ├── App.tsx             # Main app component
    │   └── main.tsx            # Entry point
    ├── package.json
    └── vite.config.ts
```

## 🛠️ API Endpoints

### Dataset Management

- `POST /dataset` - Upload a CSV dataset
- `POST /target-column` - Set target column for prediction
- `POST /model` - Select ML model

### Notebook Operations

- `GET /generate/notebook` - Generate and execute notebook (SSE stream)
- `GET /download/notebook/{session_id}` - Download generated notebook

### Model Operations

- `GET /download/model/{session_id}` - Download trained model
- `GET /download/scaler/{session_id}` - Download scaler/preprocessors
- `POST /predict/{session_id}` - Make predictions with trained model

### AI Analysis

- `GET /generate/report/{session_id}` - Generate AI-powered analysis report using LLM

### Session Management

- `GET /sessions` - List all active sessions
- `DELETE /cleanup/{session_id}` - Clean up specific session
- `DELETE /cleanup/all` - Clean up all sessions

## 🔧 Technical Stack

### Backend

- **FastAPI**: Modern, fast web framework for building APIs
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and tools
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **NBFormat**: Jupyter notebook format handling
- **NBClient**: Jupyter notebook execution
- **LangChain + Ollama**: LLM integration for AI-powered analysis reports

### Frontend

- **React 18**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing

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
