import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useTheme } from './hooks/useTheme';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Candidate from './pages/Candidate';
import Recruiter from './pages/Recruiter';

export default function App() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-white transition-colors duration-300">
        <Navbar isDark={isDark} toggleTheme={toggleTheme} />
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/candidate" element={<Candidate />} />
          <Route path="/recruiter" element={<Recruiter />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
