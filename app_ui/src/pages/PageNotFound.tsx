import "../App.css";
import { useNavigate } from "react-router-dom";

function PageNotFound() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-gradient-to-br from-black via-gray-900 to-red-950 animate-fade-in">
      <div className="flex flex-col items-center justify-center bg-black/80 border border-red-900 rounded-2xl shadow-2xl px-10 py-12 max-w-lg">
        <span className="text-7xl mb-4">🚫</span>
        <p className="text-3xl font-extrabold text-white mb-4">Whoops! Page not found!</p>
        <button
          className="mt-4 bg-gradient-to-r from-red-700 to-red-500 px-6 py-2 text-lg text-white rounded-lg shadow hover:from-red-600 hover:to-red-400 font-bold transition"
          onClick={() => navigate("/home")}
        >
          Back to Home
        </button>
      </div>
    </div>
  );
}

export default PageNotFound;