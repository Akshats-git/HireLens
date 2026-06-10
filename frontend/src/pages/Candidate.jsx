import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Send, Printer, RotateCcw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { analyzeResume } from '../services/api';
import { SingleDropZone } from '../components/DropZone';
import CircularScore from '../components/CircularScore';
import ScoreBar from '../components/ScoreBar';
import SkillTag from '../components/SkillTag';
import SuggestionCard from '../components/SuggestionCard';
import ResultsSkeleton from '../components/ResultsSkeleton';

const MIN_JD_CHARS = 50;

export default function Candidate() {
  const navigate = useNavigate();
  const [file, setFile]       = useState(null);
  const [jd, setJd]           = useState('');
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');

  const canSubmit = file && jd.trim().length >= MIN_JD_CHARS && !loading;

  const handleAnalyze = async () => {
    setError('');
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeResume(file, jd);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJd('');
    setResult(null);
    setError('');
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Page header */}
      <div className="flex items-center gap-3 mb-8 no-print">
        <button
          onClick={() => navigate('/')}
          className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <ArrowLeft size={16} />
        </button>
        <div>
          <h1 className="text-xl font-bold text-slate-900 dark:text-white">Resume Analyzer</h1>
          <p className="text-sm text-slate-400 dark:text-slate-500">See how well your resume matches a job description</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-5 gap-8">
        {/* ── Left: Input form ─────────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-5 no-print">
          {/* Resume upload */}
          <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Resume <span className="text-red-400">*</span>
            </h2>
            <SingleDropZone
              file={file}
              onDrop={setFile}
              onRemove={() => setFile(null)}
            />
          </div>

          {/* Job description */}
          <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
            <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
              Job Description <span className="text-red-400">*</span>
            </h2>
            <textarea
              value={jd}
              onChange={(e) => setJd(e.target.value)}
              placeholder="Paste the full job description here…"
              rows={9}
              className="w-full resize-none rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-sm text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition-colors"
            />
            <div className="flex justify-between mt-1.5 text-xs text-slate-400">
              <span>{jd.length} characters</span>
              {jd.length > 0 && jd.length < MIN_JD_CHARS && (
                <span className="text-amber-500">Need at least {MIN_JD_CHARS} characters</span>
              )}
            </div>
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="p-4 rounded-xl border border-red-200 dark:border-red-500/20 bg-red-50 dark:bg-red-500/5 text-sm text-red-600 dark:text-red-400"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handleAnalyze}
              disabled={!canSubmit}
              className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold text-sm transition-all shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 active:scale-95"
            >
              {loading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                />
              ) : (
                <Send size={14} />
              )}
              {loading ? 'Analyzing…' : 'Analyze Resume'}
            </button>

            {result && (
              <button
                onClick={handleReset}
                className="w-11 h-11 rounded-xl border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                title="Start over"
              >
                <RotateCcw size={14} />
              </button>
            )}
          </div>
        </div>

        {/* ── Right: Results ───────────────────────────────────────────────── */}
        <div className="lg:col-span-3">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="skeleton"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="p-6 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40"
              >
                <ResultsSkeleton />
              </motion.div>
            ) : result ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              >
                {/* Print header (only shows when printing) */}
                <div className="hidden print:block mb-6">
                  <h1 className="text-2xl font-bold">HireLens — Resume Analysis Report</h1>
                  <p className="text-sm text-gray-500">{new Date().toLocaleDateString()}</p>
                </div>

                <div id="results-panel" className="space-y-6">
                  {/* Score card */}
                  <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40 flex flex-col sm:flex-row items-center gap-6">
                    <CircularScore key={result.score} score={result.score} size={152} />
                    <div className="flex-1 space-y-1 text-center sm:text-left">
                      <p className="text-xs text-slate-400 dark:text-slate-500 uppercase tracking-wider">Overall Match</p>
                      <p className="text-4xl font-extrabold text-slate-900 dark:text-white">{result.score.toFixed(1)}</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Processed in {result.processing_time_ms?.toFixed(0)} ms
                      </p>
                    </div>
                    <button
                      onClick={() => window.print()}
                      className="no-print flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 dark:border-slate-700 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                    >
                      <Printer size={14} /> Export PDF
                    </button>
                  </div>

                  {/* Score breakdown */}
                  <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40 space-y-5">
                    <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Score Breakdown
                    </h2>
                    <div className="space-y-4" key={result.score}>
                      {Object.entries(result.breakdown).map(([key, val], i) => (
                        <ScoreBar key={key} metric={key} value={val} delay={i * 0.12} />
                      ))}
                    </div>
                  </div>

                  {/* Skills */}
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
                      <h2 className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 mb-3">
                        ✓ Matched Skills ({result.matched_skills.length})
                      </h2>
                      {result.matched_skills.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {result.matched_skills.map((s, i) => (
                            <SkillTag key={s} skill={s} variant="matched" index={i} />
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-slate-400 dark:text-slate-500">No matching skills detected.</p>
                      )}
                    </div>

                    <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
                      <h2 className="text-sm font-semibold text-red-600 dark:text-red-400 mb-3">
                        ✗ Missing Skills ({result.missing_skills.length})
                      </h2>
                      {result.missing_skills.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {result.missing_skills.map((s, i) => (
                            <SkillTag key={s} skill={s} variant="missing" index={i} />
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm text-emerald-600 dark:text-emerald-400">All required skills are present!</p>
                      )}
                    </div>
                  </div>

                  {/* Suggestions */}
                  {result.suggestions?.length > 0 && (
                    <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/40">
                      <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-4">
                        Improvement Suggestions
                      </h2>
                      <div className="space-y-2">
                        {result.suggestions.map((s, i) => (
                          <SuggestionCard key={i} text={s} index={i} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-96 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-200 dark:border-slate-700/60 text-center px-8"
              >
                <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
                  <Send size={24} className="text-slate-300 dark:text-slate-600" />
                </div>
                <h3 className="font-semibold text-slate-400 dark:text-slate-500 mb-1">No analysis yet</h3>
                <p className="text-sm text-slate-300 dark:text-slate-600 max-w-xs">
                  Upload your resume and paste a job description, then click Analyze.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
