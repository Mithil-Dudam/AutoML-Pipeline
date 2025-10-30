
import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import api from "../Api";

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
        } catch (err) {
            setError("Failed to set target column.");
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
                navigate("/results", { state: { sessionId, columns, targetColumn } });
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