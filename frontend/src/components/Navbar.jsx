import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Sun, Moon, User, Briefcase } from 'lucide-react';

export default function Navbar({ isDark, toggleTheme }) {
  const { pathname } = useLocation();

  const links = [
    { to: '/candidate', label: 'Candidate', icon: User },
    { to: '/recruiter', label: 'Recruiter', icon: Briefcase },
  ];

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center shadow-lg shadow-blue-500/30 group-hover:shadow-blue-500/50 transition-shadow">
            <Brain size={16} className="text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">
            Hire<span className="text-blue-500">Lens</span>
          </span>
        </Link>

        {/* Nav links + theme toggle */}
        <div className="flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => {
            const active = pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={`relative flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? 'text-blue-500 dark:text-blue-400'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                <Icon size={14} />
                {label}
                {active && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute inset-0 rounded-lg bg-blue-50 dark:bg-blue-500/10 -z-10"
                    transition={{ type: 'spring', duration: 0.4 }}
                  />
                )}
              </Link>
            );
          })}

          <div className="w-px h-6 bg-slate-200 dark:bg-slate-700 mx-2" />

          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="w-9 h-9 rounded-lg flex items-center justify-center text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <motion.div
              key={isDark ? 'moon' : 'sun'}
              initial={{ rotate: -30, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              transition={{ duration: 0.2 }}
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </motion.div>
          </button>
        </div>
      </nav>
    </header>
  );
}
