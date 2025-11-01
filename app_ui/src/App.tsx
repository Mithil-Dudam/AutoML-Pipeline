import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import './App.css'
import Home from './pages/Home';
import PageNotFound from './pages/PageNotFound';
import Results from './pages/Results';
import ColumnReview from './pages/ColumnReview';

function App() {
  return (
  
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/home" element={<Home />} />
          <Route path="/column-review/:sessionId" element={<ColumnReview />} />
          <Route path="/results/:sessionId" element={<Results />} />
          <Route path="*" element={<PageNotFound />} />
        </Routes>
      </Router>
    
  );
}

export default App;
