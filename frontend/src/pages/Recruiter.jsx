import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Users, SlidersHorizontal, X, Search, ChevronDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { bulkAnalyze, filterCandidates } from '../services/api';
import { MultiDropZone } from '../components/DropZone';
import CandidateSlideOver from '../components/CandidateSlideOver';
import SkillTag from '../components/SkillTag';

// ── Helpers ────────────────────────────────────────────────────────────────────

function getScoreStyle(score) {
  if (score >= 80) return { ring: 'ring-emerald-500/30', text: 'text-emerald-500', bg: 'bg-emerald-500' };
  if (score >= 60) return { ring: 'ring-amber-500/30',  text: 'text-amber-500',  bg: 'bg-amber-500'  };
  return              { ring: 'ring-red-500/30',    text: 'text-red-500',    bg: 'bg-red-500'    };
}

function ScoreBadge({ score }) {
  const { text, bg } = getScoreStyle(score);
  return (
    <div className="relative w-12 h-12 shrink-0">
      <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
        <circle cx="18" cy="18" r="15" fill="none" stroke="currentColor" strokeWidth="3" className="text-slate-200 dark:text-slate-700" />
        <motion.circle
          cx="18" cy="18" r="15"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={`${(score / 100) * 94.2} 94.2`}
          initial={{ strokeDasharray: '0 94.2' }}
          animate={{ strokeDasharray: `${(score / 100) * 94.2} 94.2` }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className={text}
        />
      </svg>
      <div className={`absolute inset-0 flex items-center justify-center text-xs font-bold ${text}`}>
        {score.toFixed(0)}
      </div>
    </div>
  );
}

function CandidateCard({ candidate, index, onViewReport }) {
  const { ring } = getScoreStyle(candidate.score);
  const name = candidate.filename.replace(/\.pdf$/i, '').replace(/[_-]/g, ' ');
  const top3 = candidate.matched_skills?.slice(0, 3) ?? [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35 }}
      className={`group flex items-center gap-4 p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40 hover:border-slate-300 dark:hover:border-slate-600 hover:shadow-md dark:hover:shadow-none transition-all duration-200 ring-0 hover:ring-2 ${ring}`}
    >
      {/* Rank */}
      <div className="w-7 h-7 rounded-full bg-slate-100 dark:bg-slate-700 flex items-center justify-center shrink-0">
        <span className="text-xs font-bold text-slate-500 dark:text-slate-400">#{candidate.rank}</span>
      </div>

      {/* Score ring */}
      <ScoreBadge score={candidate.score} />

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-sm text-slate-900 dark:text-white capitalize truncate">{name}</p>
        <div className="flex items-center gap-3 mt-1">
          {candidate.experience_level && (
            <span className="text-xs text-slate-400 dark:text-slate-500 capitalize">
              {candidate.experience_level} level
            </span>
          )}
          {top3.length > 0 && (
            <div className="flex gap-1 overflow-hidden">
              {top3.map((s) => (
                <SkillTag key={s} skill={s} variant="neutral" index={0} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Action */}
      <button
        onClick={() => onViewReport(candidate)}
        className="no-print shrink-0 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-700 hover:bg-blue-50 dark:hover:bg-blue-500/10 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
      >
        View Report
      </button>
    </motion.div>
  );
}

// ── Filter panel ───────────────────────────────────────────────────────────────

function FilterPanel({ filters, onChange, onApply, onReset, disabled }) {
  const [skillInput, setSkillInput] = useState('');

  const addSkill = (e) => {
    if ((e.key === 'Enter' || e.key === ',') && skillInput.trim()) {
      e.preventDefault();
      const s = skillInput.trim().toLowerCase();
      if (!filters.must_have_skills.includes(s)) {
        onChange({ ...filters, must_have_skills: [...filters.must_have_skills, s] });
      }
      setSkillInput('');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm text-slate-700 dark:text-slate-300 flex items-center gap-2">
          <SlidersHorizontal size={14} /> Filters
        </h3>
        <button
          onClick={onReset}
          className="text-xs text-slate-400 hover:text-slate-700 dark:hover:text-white transition-colors"
        >
          Reset
        </button>
      </div>

      {/* Min score */}
      <div>
        <div className="flex justify-between mb-2">
          <label className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Min Score</label>
          <span className="text-xs font-bold text-blue-500">{filters.min_score}</span>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={filters.min_score}
          onChange={(e) => onChange({ ...filters, min_score: Number(e.target.value) })}
          className="w-full h-1.5"
        />
        <div className="flex justify-between text-xs text-slate-300 dark:text-slate-600 mt-1">
          <span>0</span><span>50</span><span>100</span>
        </div>
      </div>

      {/* Must-have skills */}
      <div>
        <label className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
          Must-Have Skills
        </label>
        <input
          type="text"
          value={skillInput}
          onChange={(e) => setSkillInput(e.target.value)}
          onKeyDown={addSkill}
          placeholder="Type skill + Enter"
          className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition-colors"
        />
        {filters.must_have_skills.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {filters.must_have_skills.map((s) => (
              <span key={s} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs border border-blue-200 dark:border-blue-500/20">
                {s}
                <button
                  onClick={() => onChange({ ...filters, must_have_skills: filters.must_have_skills.filter(x => x !== s) })}
                  className="hover:text-red-500 transition-colors"
                >
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Experience level */}
      <div>
        <label className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider block mb-2">
          Min Experience Level
        </label>
        <div className="relative">
          <select
            value={filters.experience_level || ''}
            onChange={(e) => onChange({ ...filters, experience_level: e.target.value || null })}
            className="w-full appearance-none px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition-colors pr-8"
          >
            <option value="">Any level</option>
            <option value="entry">Entry</option>
            <option value="mid">Mid</option>
            <option value="senior">Senior</option>
            <option value="lead">Lead</option>
          </select>
          <ChevronDown size={12} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
        </div>
      </div>

      <button
        onClick={onApply}
        disabled={disabled}
        className="w-full py-2.5 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors flex items-center justify-center gap-2"
      >
        <Search size={13} /> Apply Filters
      </button>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

const DEFAULT_FILTERS = { min_score: 0, must_have_skills: [], experience_level: null };
const MIN_JD_CHARS = 50;

export default function Recruiter() {
  const navigate = useNavigate();
  const [files, setFiles]             = useState([]);
  const [jd, setJd]                   = useState('');
  const [candidates, setCandidates]   = useState([]);
  const [loading, setLoading]         = useState(false);
  const [filtering, setFiltering]     = useState(false);
  const [error, setError]             = useState('');
  const [analyzed, setAnalyzed]       = useState(false);
  const [filters, setFilters]         = useState(DEFAULT_FILTERS);
  const [selected, setSelected]       = useState(null);
  const [showFilters, setShowFilters] = useState(false);

  const addFiles = (newFiles) => {
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      const fresh = newFiles.filter((f) => !existing.has(f.name));
      return [...prev, ...fresh].slice(0, 50);
    });
  };

  const removeFile = (i) => setFiles((prev) => prev.filter((_, idx) => idx !== i));

  const canAnalyze = files.length > 0 && jd.trim().length >= MIN_JD_CHARS && !loading;

  const handleAnalyze = async () => {
    setError('');
    setLoading(true);
    try {
      const data = await bulkAnalyze(files, jd);
      setCandidates(data.candidates);
      setAnalyzed(true);
      setFilters(DEFAULT_FILTERS);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyFilters = async () => {
    setFiltering(true);
    try {
      const data = await filterCandidates(filters);
      setCandidates(data.candidates);
    } catch (err) {
      setError(err.message);
    } finally {
      setFiltering(false);
    }
  };

  const handleResetFilters = async () => {
    setFilters(DEFAULT_FILTERS);
    if (analyzed) {
      setFiltering(true);
      try {
        const data = await filterCandidates(DEFAULT_FILTERS);
        setCandidates(data.candidates);
      } catch {}
      setFiltering(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Header */}
      <div className="flex items-center gap-3 mb-8 no-print">
        <button
          onClick={() => navigate('/')}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <ArrowLeft size={16} />
        </button>
        <div>
          <h1 className="text-xl font-bold text-slate-900 dark:text-white">Recruiter Dashboard</h1>
          <p className="text-sm text-slate-400 dark:text-slate-500">Rank and filter multiple candidates against a job</p>
        </div>
      </div>

      {/* Upload panel */}
      <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40 mb-6 no-print">
        <div className="grid md:grid-cols-2 gap-5">
          <div>
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Resumes <span className="text-slate-400 font-normal">({files.length}/50)</span>
            </h2>
            <MultiDropZone files={files} onDrop={addFiles} onRemove={removeFile} />
          </div>

          <div className="flex flex-col">
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Job Description</h2>
            <textarea
              value={jd}
              onChange={(e) => setJd(e.target.value)}
              placeholder="Paste the job description here…"
              rows={6}
              className="flex-1 resize-none rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition-colors"
            />
          </div>
        </div>

        <AnimatePresence>
          {error && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mt-3 text-sm text-red-500 dark:text-red-400"
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        <div className="flex items-center justify-between mt-4">
          <button
            onClick={handleAnalyze}
            disabled={!canAnalyze}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm transition-all shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 active:scale-95"
          >
            {loading ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                />
                Analyzing {files.length} resumes…
              </>
            ) : (
              <>
                <Users size={14} /> Analyze All Resumes
              </>
            )}
          </button>

          {analyzed && (
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="md:hidden flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              <SlidersHorizontal size={13} /> Filters
            </button>
          )}
        </div>
      </div>

      {/* Results area */}
      {analyzed && (
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Filter sidebar — desktop always visible, mobile toggle */}
          <div className={`${showFilters ? 'block' : 'hidden'} md:block lg:col-span-1`}>
            <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40 sticky top-24">
              <FilterPanel
                filters={filters}
                onChange={setFilters}
                onApply={handleApplyFilters}
                onReset={handleResetFilters}
                disabled={filtering}
              />
            </div>
          </div>

          {/* Candidate list */}
          <div className="lg:col-span-3 space-y-3">
            <div className="flex items-center justify-between mb-1">
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {filtering ? 'Filtering…' : `${candidates.length} candidate${candidates.length !== 1 ? 's' : ''} found`}
              </p>
            </div>

            <AnimatePresence mode="wait">
              {filtering ? (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-20 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 animate-pulse mb-3" />
                  ))}
                </motion.div>
              ) : candidates.length === 0 ? (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-48 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-500"
                >
                  <Users size={28} className="mb-2 opacity-40" />
                  <p className="text-sm">No candidates match the current filters.</p>
                </motion.div>
              ) : (
                <motion.div key="list" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                  {candidates.map((c, i) => (
                    <CandidateCard
                      key={c.id}
                      candidate={c}
                      index={i}
                      onViewReport={setSelected}
                    />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Slide-over */}
      <AnimatePresence>
        {selected && (
          <CandidateSlideOver
            key={selected.id}
            candidate={selected}
            onClose={() => setSelected(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
