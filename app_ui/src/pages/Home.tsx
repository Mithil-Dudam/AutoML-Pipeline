
import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import api from "../Api";

interface Warning {
    type: string;
    message: string;
    severity: string;
    impact: string;
    columns?: string[];
    details?: string[];
}

interface Recommendation {
    type: string;
    message: string;
    action: string;
    benefits: string;
    details?: {
        threshold?: number;
        current_distribution?: Record<string, number>;
        proposed_distribution?: Record<string, number>;
        n_features?: number;
        recommended_max?: number;
    };
}

function Home() {
    const [file, setFile] = useState<File | null>(null);
    const [columns, setColumns] = useState<string[]>([]);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState("");
    const [targetColumn, setTargetColumn] = useState<string>("");
    const [sessionId, setSessionId] = useState<string>("");
    const [targetConfirmed, setTargetConfirmed] = useState(false);
    const [selectedModel, setSelectedModel] = useState("");
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [modelConfirmed, setModelConfirmed] = useState(false);
    const [uploadConfirmed, setUploadConfirmed] = useState(false);
    const [warnings, setWarnings] = useState<Warning[]>([]);
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [transformationApplied, setTransformationApplied] = useState<{type: string, threshold?: number} | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (targetConfirmed) return; 
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        } else {
            setFile(null);
        }
        setColumns([]);
        setError("");
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select a CSV file.");
            return;
        }
        setUploading(true);
        setError("");
        try {
            const formData = new FormData();
            formData.append("file", file);
            const response = await api.post("/dataset", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });
            setColumns(response.data.columns);
            setSessionId(response.data.session_id);
            setUploadConfirmed(true);
        } catch (err) {
            setError("Upload failed.");
        } finally {
            setUploading(false);
        }
    };

    const handleSetTargetColumn = async () => {
        if (!sessionId) {
            setError("Session ID missing. Please upload a dataset first.");
            return;
        }
        if (!targetColumn) {
            setError("Please select a target column.");
            return;
        }
        try {
            const formData = new FormData();
            formData.append("session_id", sessionId);
            formData.append("column_name", targetColumn);
            const response = await api.post("/target-column", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setTargetConfirmed(true);
            if (response.data.models) {
                setAvailableModels(response.data.models);
            } else {
                setAvailableModels([]);
            }
            // Store warnings and recommendations
            if (response.data.warnings) {
                setWarnings(response.data.warnings);
            }
            if (response.data.recommendations) {
                setRecommendations(response.data.recommendations);
            }
        } catch (err) {
            setError("Failed to set target column.");
        }
    };

    const handleApplyTransformation = async (recommendation: Recommendation) => {
        if (!sessionId) {
            setError("Session ID missing.");
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append("session_id", sessionId);
            formData.append("transformation_type", recommendation.type);
            
            if (recommendation.details?.threshold) {
                formData.append("threshold", recommendation.details.threshold.toString());
            }
            
            await api.post("/store-transformation", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            
            // Store transformation info to display confirmation
            setTransformationApplied({
                type: recommendation.type,
                threshold: recommendation.details?.threshold
            });
            
            // Keep warnings visible but clear recommendations since they're applied
            // This shows users what problem was solved
            setRecommendations([]);
            
        } catch (err) {
            setError("Failed to store transformation.");
        }
    };

    const handleSetModel = async () => {
        if (!sessionId) {
            setError("Session ID missing. Please upload a dataset first.");
            return;
        }
        if (!selectedModel) {
            setError("Please select a model.");
            return;
        }
        try {
            const formData = new FormData();
            formData.append("session_id", sessionId);
            formData.append("model_name", selectedModel);
            await api.post("/model", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            setModelConfirmed(true);
            setTimeout(() => {
                navigate(`/column-review/${sessionId}`);
            }, 1000);
        } catch (err) {
            setError("Failed to set model.");
        }
    };

    return (
        <div className="min-h-screen w-full bg-gradient-to-br from-black via-gray-900 to-blue-950 flex flex-col items-center justify-center px-4 font-sans">
            <h1 className="text-6xl font-extrabold text-white drop-shadow-lg tracking-tight mb-10 mt-10 text-center">
                <span className="border-b-4 border-blue-600 pb-2 px-4 rounded">ML Project Maker</span>
            </h1>
            <div className="bg-black/80 border border-blue-900 rounded-2xl shadow-2xl p-10 w-full max-w-xl flex flex-col gap-6 animate-fade-in">
                <label className="block text-white text-lg font-semibold mb-2">Upload Dataset</label>
                <div className="relative block w-full mb-2 flex items-center gap-2">
                    <label className="flex-1">
                        <input
                            type="file"
                            accept=".csv"
                            className="hidden"
                            ref={inputRef}
                            onChange={handleFileChange}
                            disabled={uploading || uploadConfirmed}
                        />
                        <div className="flex items-center">
                            <button
                                type="button"
                                className={`w-full bg-black/60 border border-blue-900 rounded-lg px-3 py-2 text-white text-left focus:outline-none focus:border-blue-600 focus:ring-2 focus:ring-blue-900 transition ${(uploading || uploadConfirmed) ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer hover:bg-blue-950'}`}
                                onClick={() => {
                                    if (uploading || uploadConfirmed) return;
                                    inputRef.current?.click();
                                }}
                                disabled={uploading || uploadConfirmed}
                            >
                                {file ? file.name : "Click to select a CSV file..."}
                            </button>
                        </div>
                    </label>
                </div>
                <button
                    className={`bg-gradient-to-r from-blue-700 to-blue-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-blue-600 hover:to-blue-400 font-bold transition ${(uploading || uploadConfirmed) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    onClick={handleUpload}
                    disabled={uploading || !file || uploadConfirmed}
                >
                    {uploading ? "Uploading..." : "Upload"}
                </button>
                {columns.length > 0 && (
                    <div className="mt-4">
                        <label className="block text-white text-lg font-semibold mb-2">Select the target column</label>
                        <div className="flex flex-wrap gap-4">
                            {columns.map((col) => (
                                <button
                                    key={col}
                                    type="button"
                                    className={`px-3 py-1 rounded-full border-2 font-semibold transition-all duration-150 text-white focus:outline-none focus:ring-2 focus:ring-blue-400 ${
                                        targetColumn === col
                                            ? "bg-blue-700 border-blue-700 shadow-lg"
                                            : "bg-black/60 border-blue-900 hover:bg-blue-950"
                                    }`}
                                    onClick={() => !targetConfirmed && setTargetColumn(col)}
                                    disabled={targetConfirmed}
                                >
                                    {col}
                                </button>
                            ))}
                        </div>
                        {targetColumn && (
                            <div className="text-blue-300 text-sm mt-4">Selected target: <span className="font-mono">{targetColumn}</span></div>
                        )}
                        <button
                            className={`mt-4 bg-gradient-to-r from-blue-700 to-blue-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-blue-600 hover:to-blue-400 font-bold transition ${(!targetColumn || targetConfirmed) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            onClick={handleSetTargetColumn}
                            disabled={!targetColumn || targetConfirmed}
                        >
                            Set Target Column
                        </button>
                    </div>
                )}

                {/* Warnings Section */}
                {warnings.length > 0 && (
                    <div className={`mt-6 p-5 border-2 rounded-xl shadow animate-fade-in ${
                        transformationApplied 
                            ? 'bg-green-900/30 border-green-600' 
                            : 'bg-yellow-900/30 border-yellow-600'
                    }`}>
                        <h3 className={`text-xl font-bold mb-3 flex items-center gap-2 ${
                            transformationApplied ? 'text-green-300' : 'text-yellow-300'
                        }`}>
                            {transformationApplied ? (
                                <>
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    ✅ Issues Being Addressed by Transformation
                                </>
                            ) : (
                                <>
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                    </svg>
                                    ⚠️ Dataset Warnings
                                </>
                            )}
                        </h3>
                        {transformationApplied && (
                            <div className="mb-4 p-3 bg-green-950/50 border border-green-700 rounded-lg">
                                <p className="text-green-200 text-sm font-semibold flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                    </svg>
                                    Binary transformation applied - these warnings will be resolved during training
                                </p>
                            </div>
                        )}
                        {warnings.map((warning, index) => (
                            <div key={index} className="mb-3 p-3 bg-black/50 rounded-lg border border-yellow-700">
                                <div className="flex items-start gap-2">
                                    <span className={`text-xs font-bold px-2 py-1 rounded ${
                                        warning.severity === 'high' ? 'bg-red-600 text-white' : 
                                        warning.severity === 'medium' ? 'bg-orange-600 text-white' : 
                                        'bg-yellow-600 text-white'
                                    }`}>
                                        {warning.severity.toUpperCase()}
                                    </span>
                                    <div className="flex-1">
                                        <p className="text-yellow-100 font-semibold text-sm">{warning.message}</p>
                                        <p className="text-yellow-200/70 text-xs mt-1">{warning.impact}</p>
                                        
                                        {/* Show detailed breakdown if available */}
                                        {warning.details && Array.isArray(warning.details) && (
                                            <div className="mt-2 space-y-1">
                                                {warning.details.map((detail: string, idx: number) => (
                                                    <div key={idx} className="text-xs text-yellow-100 font-mono bg-yellow-900/30 px-2 py-1 rounded">
                                                        {detail}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        
                                        {/* Show column badges if no details */}
                                        {!warning.details && warning.columns && warning.columns.length > 0 && (
                                            <div className="mt-2 flex flex-wrap gap-1">
                                                {warning.columns.map((col) => (
                                                    <span key={col} className="text-xs bg-yellow-900/50 text-yellow-200 px-2 py-0.5 rounded font-mono">
                                                        {col}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Recommendations Section */}
                {recommendations.length > 0 && (
                    <div className="mt-6 p-5 bg-blue-900/30 border-2 border-blue-600 rounded-xl shadow animate-fade-in">
                        <h3 className="text-xl font-bold text-blue-300 mb-3 flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                            💡 Smart Recommendations
                        </h3>
                        {recommendations.map((rec, index) => (
                            <div key={index} className="mb-4 p-4 bg-black/50 rounded-lg border border-blue-700">
                                <div className="mb-2">
                                    <p className="text-blue-100 font-semibold text-base">{rec.message}</p>
                                    <p className="text-green-300 text-sm mt-1">✓ {rec.benefits}</p>
                                </div>
                                
                                {rec.type === 'binary_transformation' && rec.details?.threshold && (
                                    <div className="mt-3 p-3 bg-blue-950/50 rounded border border-blue-800">
                                        <p className="text-blue-200 text-sm font-mono mb-2">
                                            Suggested: Values ≥ {rec.details.threshold} → Class 1, Values &lt; {rec.details.threshold} → Class 0
                                        </p>
                                        {rec.details.current_distribution && rec.details.proposed_distribution && (
                                            <div className="grid grid-cols-2 gap-3 mt-2">
                                                <div>
                                                    <p className="text-xs text-blue-300 font-semibold mb-1">Current:</p>
                                                    {Object.entries(rec.details.current_distribution).map(([key, value]) => (
                                                        <p key={key} className="text-xs text-blue-200">
                                                            Class {key}: {value} samples
                                                        </p>
                                                    ))}
                                                </div>
                                                <div>
                                                    <p className="text-xs text-green-300 font-semibold mb-1">After Transform:</p>
                                                    {Object.entries(rec.details.proposed_distribution).map(([key, value]) => (
                                                        <p key={key} className="text-xs text-green-200">
                                                            {key}: {value} samples
                                                        </p>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                                
                                {rec.type === 'feature_reduction' && rec.details?.n_features && rec.details?.recommended_max && (
                                    <div className="mt-3 p-3 bg-blue-950/50 rounded border border-blue-800">
                                        <p className="text-blue-200 text-sm">
                                            Current: {rec.details.n_features} features → Recommended: {rec.details.recommended_max} features
                                        </p>
                                    </div>
                                )}
                                
                                {/* Apply Transformation Button */}
                                {rec.type === 'binary_transformation' && (
                                    <button
                                        className="mt-4 w-full bg-gradient-to-r from-green-600 to-green-500 px-4 py-3 text-base text-white rounded-lg shadow-lg hover:from-green-500 hover:to-green-400 font-bold transition flex items-center justify-center gap-2"
                                        onClick={() => handleApplyTransformation(rec)}
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                        Apply Binary Transformation
                                    </button>
                                )}
                            </div>
                        ))}
                        
                        {/* Optional Note */}
                        <div className="mt-4 p-3 bg-blue-950/30 border border-blue-700/50 rounded-lg">
                            <p className="text-blue-200 text-sm flex items-start gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span>
                                    <strong>Note:</strong> These transformations are optional suggestions to improve model performance. 
                                    You can proceed without applying them by selecting a model below.
                                </span>
                            </p>
                        </div>
                    </div>
                )}

                {/* Transformation Applied Confirmation */}
                {transformationApplied && (
                    <div className="mt-6 p-5 bg-green-900/30 border-2 border-green-500 rounded-xl shadow animate-fade-in">
                        <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 mt-1">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <div className="flex-1">
                                <h3 className="text-lg font-bold text-green-300 mb-1">✅ Transformation Accepted!</h3>
                                {transformationApplied.type === 'binary_transformation' && transformationApplied.threshold && (
                                    <p className="text-green-100 text-sm">
                                        Binary transformation will be applied in the notebook: 
                                        <span className="font-mono bg-green-950/50 px-2 py-1 rounded ml-2">
                                            Values ≥ {transformationApplied.threshold} → Class 1, Values &lt; {transformationApplied.threshold} → Class 0
                                        </span>
                                    </p>
                                )}
                                <p className="text-green-200/80 text-xs mt-2">
                                    The transformation will be documented and applied when you generate the notebook.
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {targetConfirmed && availableModels.length > 0 && (
                    <div className="mt-8 p-6 bg-black/70 border border-blue-800 rounded-xl shadow flex flex-col gap-4 animate-fade-in">
                        <label className="block text-white text-lg font-semibold mb-2">Select a Machine Learning Model</label>
                        <div className="flex flex-wrap gap-4">
                            {availableModels.map((model) => (
                                <button
                                    key={model}
                                    type="button"
                                    className={`px-4 py-2 rounded-lg border-2 font-semibold transition-all duration-150 text-white focus:outline-none focus:ring-2 focus:ring-blue-400 ${
                                        selectedModel === model
                                            ? "bg-blue-700 border-blue-700 shadow-lg"
                                            : "bg-black/60 border-blue-900 hover:bg-blue-950"
                                    }`}
                                    onClick={() => !modelConfirmed && setSelectedModel(model)}
                                    disabled={modelConfirmed}
                                >
                                    {model}
                                </button>
                            ))}
                        </div>
                        {selectedModel && (
                            <div className="text-blue-300 text-sm mt-4">Selected model: <span className="font-mono">{selectedModel}</span></div>
                        )}
                        <button
                            className={`mt-4 bg-gradient-to-r from-blue-700 to-blue-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-blue-600 hover:to-blue-400 font-bold transition ${(!selectedModel || modelConfirmed) ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            onClick={handleSetModel}
                            disabled={!selectedModel || modelConfirmed}
                        >
                            Set Model
                        </button>
                        {modelConfirmed && (
                            <div className="text-green-400 text-sm mt-4">Model set successfully!</div>
                        )}
                    </div>
                )}
                {error && <p className="text-red-400 text-center mt-4 font-semibold">{error}</p>}
                {uploading && (
                    <div className="flex justify-center items-center my-4">
                        <svg
                            className="animate-spin h-6 w-6 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                            ></circle>
                            <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8v8H4z"
                            ></path>
                        </svg>
                        <p className="ml-2 text-white">Processing...</p>
                    </div>
                )}
            </div>
            <footer className="mt-12 text-blue-200 text-xs opacity-80">&copy; {new Date().getFullYear()} ML Project Maker</footer>
            <style>{`
                @keyframes fade-in { from { opacity: 0; transform: translateY(30px);} to { opacity: 1; transform: none; } }
                .animate-fade-in { animation: fade-in 0.8s cubic-bezier(.4,0,.2,1) both; }
            `}</style>
        </div>
    );
}

export default Home;