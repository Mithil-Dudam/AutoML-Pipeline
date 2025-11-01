import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

function Results() {
    const { sessionId } = useParams<{ sessionId: string }>();
    const [results, setResults] = useState<Array<{ cell: number; result: string | null }>>([]);
    const [loading, setLoading] = useState(true);
    const [initializing, setInitializing] = useState(true);
    const [error, setError] = useState("");
    const [testInputs, setTestInputs] = useState<Record<string, string>>({});
    const [prediction, setPrediction] = useState<any>(null);
    const [predicting, setPredicting] = useState(false);
    const [columns, setColumns] = useState<string[]>([]);
    const [categoricalValues, setCategoricalValues] = useState<Record<string, string[]>>({});
    const [aiReport, setAiReport] = useState<string>("");
    const [generatingReport, setGeneratingReport] = useState(false);
    const [reportEventSource, setReportEventSource] = useState<EventSource | null>(null);

    useEffect(() => {
        if (!sessionId) {
            setError("Session ID missing. Please return to the previous step.");
            setLoading(false);
            return;
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
                    // Hide initializing spinner once first result arrives
                    if (initializing) {
                        setInitializing(false);
                    }
                } catch (e) {
                    setResults((prev) => [...prev, { cell: results.length + 1, result: event.data }]);
                    if (initializing) {
                        setInitializing(false);
                    }
                }
            };
            eventSource.onerror = () => {
                setLoading(false);
                eventSource && eventSource.close();
                
                // Fetch the actual input columns needed for prediction (after preprocessing)
                fetch(`http://localhost:8000/input-columns/${sessionId}`)
                    .then(res => res.json())
                    .then(data => {
                        if (data.input_columns) {
                            setColumns(data.input_columns);
                        }
                    })
                    .catch(err => {
                        console.error("Failed to fetch input columns:", err);
                    });
                
                // Fetch categorical values for dropdowns
                fetch(`http://localhost:8000/categorical-values/${sessionId}`)
                    .then(res => res.json())
                    .then(data => {
                        if (data.categorical_values) {
                            setCategoricalValues(data.categorical_values);
                        }
                    })
                    .catch(err => {
                        console.error("Failed to fetch categorical values:", err);
                    });
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

    // Cleanup report EventSource on unmount
    useEffect(() => {
        return () => {
            if (reportEventSource) reportEventSource.close();
        };
    }, [reportEventSource]);

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
                {initializing && (
                    <div className="bg-black/70 border border-blue-800 rounded-xl p-8 shadow-2xl flex flex-col items-center justify-center gap-4">
                        <svg className="animate-spin h-12 w-12 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <p className="text-white text-xl font-semibold">Generating Notebook & Training Model...</p>
                        <p className="text-blue-300 text-sm">This may take a few moments</p>
                    </div>
                )}
                {error && <p className="text-red-400 font-semibold">{error}</p>}
                {!initializing && results.filter(res => res.cell > 3).map((res, idx) => {
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
                        <div className="bg-black/80 border border-cyan-900 rounded-2xl shadow-2xl p-8 w-full flex flex-col gap-4 mt-6">
                            <div className="text-cyan-200 text-xl font-bold mb-2">🤖 AI-Generated Analysis Report</div>
                            <p className="text-gray-300 text-sm mb-4">
                                Get an expert AI analysis of your ML pipeline results with actionable insights and recommendations.
                            </p>
                            
                            {!aiReport && (
                                <button
                                    onClick={async () => {
                                        setGeneratingReport(true);
                                        try {
                                            // Start report generation
                                            await fetch(`http://localhost:8000/generate/report/${sessionId}`);
                                            
                                            // Use Server-Sent Events for real-time updates (more efficient than polling)
                                            const eventSource = new EventSource(`http://localhost:8000/report/stream/${sessionId}`);
                                            setReportEventSource(eventSource);
                                            
                                            eventSource.onmessage = (event) => {
                                                try {
                                                    const data = JSON.parse(event.data);
                                                    
                                                    if (data.status === "completed") {
                                                        setAiReport(data.report);
                                                        setGeneratingReport(false);
                                                        eventSource.close();
                                                        setReportEventSource(null);
                                                    } else if (data.status === "failed") {
                                                        setAiReport("Failed to generate report: " + (data.error || "Unknown error"));
                                                        setGeneratingReport(false);
                                                        eventSource.close();
                                                        setReportEventSource(null);
                                                    }
                                                    // Status "generating" - keep spinner visible
                                                } catch (parseErr) {
                                                    console.error("Error parsing SSE data:", parseErr);
                                                }
                                            };
                                            
                                            eventSource.onerror = (err) => {
                                                console.error("SSE error:", err);
                                                setAiReport("Connection error. Please try again.");
                                                setGeneratingReport(false);
                                                eventSource.close();
                                                setReportEventSource(null);
                                            };
                                            
                                        } catch (err) {
                                            setAiReport("Failed to start report generation. Please try again.");
                                            setGeneratingReport(false);
                                        }
                                    }}
                                    disabled={generatingReport}
                                    className="bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-700 hover:to-cyan-800 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transition-all duration-300 flex items-center justify-center gap-3"
                                >
                                    {generatingReport ? (
                                        <>
                                            <span className="inline-block animate-spin">⏳</span>
                                            <span>Generating Report...</span>
                                            <span className="inline-block animate-pulse">🤖</span>
                                        </>
                                    ) : (
                                        <>
                                            <span>Generate AI Report</span>
                                            <span>🚀</span>
                                        </>
                                    )}
                                </button>
                            )}

                            {aiReport && (
                                <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-8 overflow-auto max-h-[600px] shadow-2xl">
                                    <div className="space-y-2">
                                        {aiReport.split('\n').map((line, idx) => {
                                            const trimmed = line.trim();
                                            
                                            // Empty line - add spacing
                                            if (!trimmed) {
                                                return <div key={idx} className="h-6" />;
                                            }
                                            
                                            // Heading (line that ends with just ** on both sides or is bold standalone)
                                            if (/^\*\*[^*]+\*\*$/.test(trimmed)) {
                                                const headingText = trimmed.replace(/\*\*/g, '');
                                                return (
                                                    <div key={idx} className="mt-8 mb-4 pb-3 border-b border-cyan-500/30">
                                                        <h2 className="text-2xl font-bold text-cyan-400">
                                                            {headingText}
                                                        </h2>
                                                    </div>
                                                );
                                            }
                                            
                                            // Bullet point
                                            if (trimmed.startsWith('*') || trimmed.startsWith('-') || trimmed.startsWith('•')) {
                                                const content = trimmed.replace(/^[\*\-•]\s*/, '');
                                                // Handle nested bullets (with +)
                                                const isNested = content.startsWith('+');
                                                const finalContent = isNested ? content.replace(/^\+\s*/, '') : content;
                                                
                                                // Replace **text** with bold
                                                const parts = finalContent.split(/(\*\*[^*]+\*\*)/g);
                                                
                                                return (
                                                    <div key={idx} className={`flex gap-3 ${isNested ? 'ml-12' : 'ml-6'} text-gray-100 text-lg leading-relaxed mb-2`}>
                                                        <span className="text-cyan-400 flex-shrink-0 mt-1">•</span>
                                                        <div className="flex-1">
                                                            {parts.map((part, i) => {
                                                                if (part.startsWith('**') && part.endsWith('**')) {
                                                                    return <strong key={i} className="text-cyan-300 font-bold">{part.slice(2, -2)}</strong>;
                                                                }
                                                                return <span key={i}>{part}</span>;
                                                            })}
                                                        </div>
                                                    </div>
                                                );
                                            }
                                            
                                            // Regular paragraph
                                            const parts = trimmed.split(/(\*\*[^*]+\*\*)/g);
                                            return (
                                                <p key={idx} className="text-gray-100 text-lg leading-relaxed mb-3">
                                                    {parts.map((part, i) => {
                                                        if (part.startsWith('**') && part.endsWith('**')) {
                                                            return <strong key={i} className="text-cyan-300 font-bold">{part.slice(2, -2)}</strong>;
                                                        }
                                                        return <span key={i}>{part}</span>;
                                                    })}
                                                </p>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </div>

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
                                {columns.map((col) => {
                                    const isCategorical = categoricalValues[col] && categoricalValues[col].length > 0;
                                    
                                    return (
                                        <div key={col} className="flex flex-col gap-2">
                                            <label className="text-gray-300 text-sm font-semibold">{col}</label>
                                            {isCategorical ? (
                                                <select
                                                    value={testInputs[col] || ""}
                                                    onChange={(e) => setTestInputs({ ...testInputs, [col]: e.target.value })}
                                                    className="bg-gray-800 text-white border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-yellow-500 transition-colors"
                                                >
                                                    <option value="" disabled>Select {col}</option>
                                                    {categoricalValues[col].map((value) => (
                                                        <option key={value} value={value}>
                                                            {value}
                                                        </option>
                                                    ))}
                                                </select>
                                            ) : (
                                                <input
                                                    type="text"
                                                    value={testInputs[col] || ""}
                                                    onChange={(e) => setTestInputs({ ...testInputs, [col]: e.target.value })}
                                                    className="bg-gray-800 text-white border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-yellow-500 transition-colors"
                                                    placeholder={`Enter ${col}`}
                                                />
                                            )}
                                        </div>
                                    );
                                })}
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
                                disabled={
                                    predicting || 
                                    columns.length === 0 || 
                                    columns.some(col => !testInputs[col] || testInputs[col].trim() === "")
                                }
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
                                                Prediction: <span className="text-2xl text-green-400">
                                                    {typeof prediction.prediction === 'number' 
                                                        ? prediction.prediction.toFixed(4) 
                                                        : prediction.prediction}
                                                </span>
                                            </div>
                                            {prediction.probability && typeof prediction.probability === 'object' && (
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
