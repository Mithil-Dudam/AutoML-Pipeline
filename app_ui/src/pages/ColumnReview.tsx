import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";

interface ColumnInfo {
  name: string;
  dtype: string;
  unique_count: number;
  sample_values: string[];
  is_auto_excluded: boolean;
  exclusion_reason?: string;
}

const ColumnReview = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/column-info/${sessionId}`);
        const columnData: ColumnInfo[] = response.data.columns;
        setColumns(columnData);
        
        // Initialize selected columns (all columns that are NOT auto-excluded)
        const initialSelected = new Set<string>(
          columnData.filter(col => !col.is_auto_excluded).map(col => col.name)
        );
        setSelectedColumns(initialSelected);
        setLoading(false);
      } catch (err) {
        setError("Failed to load column information");
        setLoading(false);
      }
    };

    fetchColumns();
  }, [sessionId]);

  const toggleColumn = (columnName: string) => {
    const newSelected = new Set(selectedColumns);
    if (newSelected.has(columnName)) {
      newSelected.delete(columnName);
    } else {
      newSelected.add(columnName);
    }
    setSelectedColumns(newSelected);
  };

  const handleContinue = async () => {
    try {
      // Get list of columns to exclude (those NOT selected)
      const allColumnNames = columns.map(col => col.name);
      const excludedColumns = allColumnNames.filter(name => !selectedColumns.has(name));
      
      const formData = new FormData();
      formData.append("session_id", sessionId || "");
      formData.append("excluded_columns", JSON.stringify(excludedColumns));
      
      await axios.post(`http://localhost:8000/exclude-columns`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      navigate(`/results/${sessionId}`);
    } catch (err) {
      setError("Failed to save column selections");
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <div className="text-white text-xl">Loading columns...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center">
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6 max-w-md">
          <p className="text-red-400 text-xl">⚠️ {error}</p>
        </div>
      </div>
    );
  }

  const selectedCount = selectedColumns.size;
  const totalCount = columns.length;

  return (
  <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header with animated gradient */}
        <div className="mb-10 text-center">
          <div className="inline-block mb-5">
            <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-5 rounded-2xl shadow-2xl shadow-cyan-500/40 border border-cyan-500/30">
              <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
              </svg>
            </div>
          </div>
          <h1 className="text-5xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent mb-4 drop-shadow-lg">
            Review Columns
          </h1>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Select which columns to include in model training. We've automatically detected and excluded IDs, emails, and other non-predictive columns.
          </p>
        </div>

        {/* Summary Card with glassmorphism */}
  <div className="bg-black/80 border border-blue-900 rounded-2xl shadow-2xl p-8 mb-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-4 rounded-xl shadow-lg shadow-cyan-500/50 border border-cyan-500/30">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <p className="text-gray-400 text-sm mb-1">Selected Columns</p>
                <p className="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent drop-shadow">
                  {selectedCount} <span className="text-2xl text-gray-500">/ {totalCount}</span>
                </p>
              </div>
            </div>
            <button
              onClick={handleContinue}
              disabled={selectedCount === 0}
              className={`group px-8 py-4 rounded-xl font-semibold transition-all duration-300 flex items-center gap-2 ${
                selectedCount === 0
                  ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:shadow-2xl hover:shadow-cyan-500/50 hover:scale-105 border border-cyan-500/30"
              }`}
            >
              Continue to Training
              <svg className={`w-5 h-5 transition-transform ${selectedCount > 0 ? 'group-hover:translate-x-1' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </button>
          </div>
        </div>

        {/* Columns Grid */}
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {columns.map((column) => {
            const isSelected = selectedColumns.has(column.name);
            const isAutoExcluded = column.is_auto_excluded;

            return (
              <div
                key={column.name}
                className={`group relative bg-black/80 border rounded-2xl p-6 transition-all duration-300 cursor-pointer overflow-hidden ${
                  isSelected
                    ? "border-cyan-500/50 shadow-2xl shadow-cyan-500/30 scale-[1.03]"
                    : "border-gray-700/50 hover:border-cyan-500/30 hover:shadow-xl hover:scale-[1.01]"
                }`}
                style={{ boxShadow: isSelected ? '0 4px 32px 0 rgba(34,211,238,0.15), 0 1.5px 8px 0 rgba(59,130,246,0.10)' : undefined }}
                onClick={() => toggleColumn(column.name)}
              >
                {/* Animated gradient overlay */}
                {isSelected && (
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 animate-pulse pointer-events-none"></div>
                )}

                <div className="relative flex items-start gap-4">
                  {/* Custom Checkbox */}
                  <div className="flex-shrink-0 mt-1">
                    <div className={`w-6 h-6 rounded-lg border-2 flex items-center justify-center transition-all duration-300 ${
                      isSelected 
                        ? "bg-gradient-to-br from-cyan-500 to-blue-600 border-transparent shadow-lg shadow-cyan-500/50" 
                        : "border-gray-600 group-hover:border-cyan-500/50"
                    }`}>
                      {isSelected && (
                        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                  </div>

                  <div className="flex-1 min-w-0">
                    {/* Column Name and Badge */}
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-white font-semibold font-mono text-base truncate">{column.name}</h3>
                      {isAutoExcluded && (
                        <span className="flex-shrink-0 text-xs bg-yellow-900/30 text-yellow-400 px-2 py-1 rounded-md border border-yellow-500/30">
                          🔍 {column.exclusion_reason || "Auto"}
                        </span>
                      )}
                    </div>

                    {/* Stats with icons */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-lg p-2 border border-cyan-500/10">
                        <div className="flex items-center gap-1 mb-1">
                          <svg className="w-3 h-3 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                          </svg>
                          <p className="text-gray-400 text-xs">Type</p>
                        </div>
                        <p className="text-gray-200 text-xs font-mono font-semibold">{column.dtype}</p>
                      </div>
                      <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-lg p-2 border border-cyan-500/10">
                        <div className="flex items-center gap-1 mb-1">
                          <svg className="w-3 h-3 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
                          </svg>
                          <p className="text-gray-400 text-xs">Unique</p>
                        </div>
                        <p className="text-gray-200 text-xs font-mono font-semibold">{column.unique_count.toLocaleString()}</p>
                      </div>
                    </div>

                    {/* Sample Values with gradient tags */}
                    <div>
                      <p className="text-gray-400 text-xs mb-2 flex items-center gap-1">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Sample Values
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {column.sample_values.slice(0, 3).map((value, idx) => (
                          <span
                            key={idx}
                            className="text-xs bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-200 px-2.5 py-1 rounded-md font-mono border border-cyan-500/20"
                          >
                            {String(value).length > 15 ? String(value).slice(0, 15) + '...' : value}
                          </span>
                        ))}
                        {column.sample_values.length > 3 && (
                          <span className="text-xs text-gray-400 px-2 py-1">
                            +{column.sample_values.length - 3}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ColumnReview;
