import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

function Results() {
    const location = useLocation();
    const [results, setResults] = useState<Array<{ cell: number; result: string | null }>>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [testInputs, setTestInputs] = useState<Record<string, string>>({});
    const [prediction, setPrediction] = useState<any>(null);
    const [predicting, setPredicting] = useState(false);
    const [columns, setColumns] = useState<string[]>([]);

    const sessionId = location.state?.sessionId;
    const targetColumn = location.state?.targetColumn;

    useEffect(() => {
        if (!sessionId) {
            setError("Session ID missing. Please return to the previous step.");
            setLoading(false);
            return;
        }
        
        // Get columns from location state
        if (location.state?.columns) {
            const cols = location.state.columns.filter((col: string) => col !== targetColumn);
            setColumns(cols);
        }
        
        let eventSource: EventSource | null = null;
        setResults([]);
        setLoading(true);
        setError("");
        try {
            eventSource = new EventSource(`http://localhost:8000/generate/notebook?session_id=${sessionId}`);
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setResults((prev) => [...prev, { cell: data.cell, result: data.result }]);
                } catch (e) {
                    setResults((prev) => [...prev, { cell: results.length + 1, result: event.data }]);
                }
            };
            eventSource.onerror = () => {
                setLoading(false);
                eventSource && eventSource.close();
            };
            eventSource.onopen = () => setLoading(false);
        } catch (err) {
            setError("Failed to fetch notebook results.");
            setLoading(false);
        }
        return () => {
            if (eventSource) eventSource.close();
        };
    }, [sessionId]);

    const cellTitles = [
        "First 5 Rows of Dataset",
        "Dataset Info",
        "Summary Statistics",
        "Target Column Distribution",
        "Missing Values Check",
        "Missing Value Imputation",
        "Feature Engineering",
        "Train-Test Split",
        "K-Fold Cross-Validation",
        "Final Model Training and Evaluation"
    ];

    return (
        <div className="min-h-screen w-full bg-gradient-to-br from-black via-gray-900 to-blue-950 flex flex-col items-center justify-center px-4 font-sans">
            <h1 className="text-4xl font-extrabold text-white drop-shadow-lg tracking-tight mb-8 mt-10 text-center">
                <span className="border-b-4 border-blue-600 pb-2 px-4 rounded">Notebook Results</span>
            </h1>
            <div className="w-full max-w-4xl flex flex-col gap-8 animate-fade-in">
                {loading && <p className="text-white">Loading results...</p>}
                {error && <p className="text-red-400 font-semibold">{error}</p>}
                {results.filter(res => res.cell > 3).map((res, idx) => {
                    // Only treat as HTML if it starts with a real HTML tag (e.g., <table, <div, <style, etc.)
                    const htmlTagPattern = /^<(table|div|style|thead|tbody|tr|th|td|html|span|p|ul|ol|li|h[1-6]|br|hr|img|section|article|header|footer|nav|main|form|input|button|label|select|option|textarea|pre|code|blockquote|figure|figcaption|canvas|svg|math)/i;
                    const isHtml = res.result && htmlTagPattern.test(res.result.trim());
                    return (
                        <div key={idx} className="bg-black/80 border border-blue-900 rounded-2xl shadow-2xl p-8 w-full flex flex-col gap-4">
                            <div className="text-blue-200 text-lg font-semibold mb-2">{cellTitles[res.cell - 4] || `Cell ${res.cell}`}</div>
                            <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto w-full">
                                <style>{`
                                    .notebook-result-table {
                                        color: #f1f1f1 !important;
                                    }
                                    .notebook-result-table table, .notebook-result-table th, .notebook-result-table td {
                                        background: #181e2a !important;
                                        color: #f1f1f1 !important;
                                        border-color: #334155 !important;
                                    }
                                    .notebook-result-table tr { background: #181e2a !important; }
                                    .notebook-result-table thead th { font-weight: bold; }
                                    .notebook-result-table table {
                                        border-collapse: separate !important;
                                        border-spacing: 0 !important;
                                        width: 100% !important;
                                    }
                                    .notebook-result-table th, .notebook-result-table td {
                                        padding: 0.75rem 1.25rem !important;
                                        text-align: left !important;
                                        font-size: 1rem !important;
                                        vertical-align: middle !important;
                                    }
                                `}</style>
                                {isHtml ? (
                                    <div className="notebook-result-table text-gray-100" dangerouslySetInnerHTML={{ __html: res.result || "" }} />
                                ) : (
                                    <pre className="text-slate-100 font-mono whitespace-pre-wrap" style={{ fontFamily: 'Fira Mono, Menlo, Monaco, Consolas, monospace', background: 'none', margin: 0, padding: 0 }}>
                                        {res.result}
                                    </pre>
                                )}
                            </div>
                        </div>
                    );
                })}
                
                {!loading && results.filter(res => res.cell > 3).length >= 10 && (
                    <>
                        <div className="bg-black/80 border border-green-900 rounded-2xl shadow-2xl p-8 w-full flex flex-col gap-4 mt-6">
                            <div className="text-green-200 text-xl font-bold mb-2">Download Artifacts</div>
                            <p className="text-gray-300 text-sm mb-4">
                                Download your trained model, preprocessing artifacts, and notebook for future use.
                            </p>
                            <div className="flex flex-wrap gap-4">
                                <a
                                    href={`http://localhost:8000/download/model/${sessionId}`}
                                    download
                                    className="flex-1 min-w-[200px] bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 text-center"
                                >
                                    📦 Download Model
                                </a>
                                <a
                                    href={`http://localhost:8000/download/scaler/${sessionId}`}
                                    download
                                    className="flex-1 min-w-[200px] bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 text-center"
                                >
                                    ⚙️ Download Scaler
                                </a>
                                <a
                                    href={`http://localhost:8000/download/notebook/${sessionId}`}
                                    download
                                    className="flex-1 min-w-[200px] bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 text-center"
                                >
                                    📓 Download Notebook
                                </a>
                            </div>
                        </div>

                        <div className="bg-black/80 border border-yellow-900 rounded-2xl shadow-2xl p-8 w-full flex flex-col gap-4 mt-6">
                            <div className="text-yellow-200 text-xl font-bold mb-2">🔮 Test Your Model</div>
                            <p className="text-gray-300 text-sm mb-4">
                                Enter values for each feature to get a prediction from your trained model.
                            </p>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {columns.map((col) => (
                                    <div key={col} className="flex flex-col gap-2">
                                        <label className="text-gray-300 text-sm font-semibold">{col}</label>
                                        <input
                                            type="text"
                                            value={testInputs[col] || ""}
                                            onChange={(e) => setTestInputs({ ...testInputs, [col]: e.target.value })}
                                            className="bg-gray-800 text-white border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-yellow-500 transition-colors"
                                            placeholder={`Enter ${col}`}
                                        />
                                    </div>
                                ))}
                            </div>

                            <button
                                onClick={async () => {
                                    setPredicting(true);
                                    setPrediction(null);
                                    try {
                                        const response = await fetch(`http://localhost:8000/predict/${sessionId}`, {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify(testInputs)
                                        });
                                        const data = await response.json();
                                        setPrediction(data);
                                    } catch (err) {
                                        setPrediction({ error: "Prediction failed. Please try again." });
                                    } finally {
                                        setPredicting(false);
                                    }
                                }}
                                disabled={predicting || columns.length === 0}
                                className="bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 mt-4"
                            >
                                {predicting ? "Predicting..." : "Get Prediction"}
                            </button>

                            {prediction && (
                                <div className={`mt-4 p-6 rounded-lg ${prediction.error ? "bg-red-900/30 border border-red-700" : "bg-green-900/30 border border-green-700"}`}>
                                    {prediction.error ? (
                                        <p className="text-red-300 font-semibold">{prediction.error}</p>
                                    ) : (
                                        <>
                                            <div className="text-green-200 text-lg font-bold mb-2">
                                                Prediction: <span className="text-2xl text-green-400">{prediction.prediction.toFixed(4)}</span>
                                            </div>
                                            {prediction.probability && (
                                                <div className="mt-3">
                                                    <p className="text-gray-300 text-sm font-semibold mb-2">Class Probabilities:</p>
                                                    {Object.entries(prediction.probability).map(([cls, prob]: [string, any]) => (
                                                        <div key={cls} className="flex justify-between text-gray-300 text-sm">
                                                            <span>Class {cls}:</span>
                                                            <span className="font-semibold">{(prob * 100).toFixed(2)}%</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

export default Results;
