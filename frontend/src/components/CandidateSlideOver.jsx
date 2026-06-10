import { motion, AnimatePresence } from 'framer-motion';
import { X, Download } from 'lucide-react';
import CircularScore from './CircularScore';
import ScoreBar from './ScoreBar';
import SkillTag from './SkillTag';
import SuggestionCard from './SuggestionCard';

export default function CandidateSlideOver({ candidate, onClose }) {
  if (!candidate) return null;

  const candidateName = candidate.filename.replace(/\.pdf$/i, '').replace(/[_-]/g, ' ');

  const handlePrint = () => window.print();

  return (
    <AnimatePresence>
      <>
        {/* Backdrop */}
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          onClick={onClose}
        />

        {/* Panel */}
        <motion.aside
          key="panel"
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 28, stiffness: 280 }}
          className="fixed right-0 top-0 h-full w-full max-w-xl z-50 flex flex-col bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700 shadow-2xl overflow-hidden"
        >
          {/* Header */}
          <div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-700">
            <div>
              <p className="text-xs text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-0.5">Full Report</p>
              <h2 className="font-semibold text-slate-900 dark:text-white capitalize truncate max-w-64">
                {candidateName}
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handlePrint}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                title="Export PDF"
              >
                <Download size={15} />
              </button>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Scrollable body */}
          <div className="flex-1 overflow-y-auto px-6 py-6 space-y-8">
            {/* Score ring */}
            <div className="flex justify-center">
              <CircularScore key={candidate.id} score={candidate.score} size={152} />
            </div>

            {/* Rank badge */}
            <div className="flex justify-center">
              <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
                Rank #{candidate.rank} among candidates
              </span>
            </div>

            {/* Score breakdown */}
            <section>
              <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-4">
                Score Breakdown
              </h3>
              <div className="space-y-4" key={candidate.id}>
                {Object.entries(candidate.breakdown).map(([key, val], i) => (
                  <ScoreBar key={key} metric={key} value={val} delay={i * 0.1} />
                ))}
              </div>
            </section>

            {/* Matched skills */}
            {candidate.matched_skills?.length > 0 && (
              <section>
                <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
                  Matched Skills ({candidate.matched_skills.length})
                </h3>
                <div className="flex flex-wrap gap-2">
                  {candidate.matched_skills.map((s, i) => (
                    <SkillTag key={s} skill={s} variant="matched" index={i} />
                  ))}
                </div>
              </section>
            )}

            {/* Missing skills */}
            {candidate.missing_skills?.length > 0 && (
              <section>
                <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
                  Missing Skills ({candidate.missing_skills.length})
                </h3>
                <div className="flex flex-wrap gap-2">
                  {candidate.missing_skills.map((s, i) => (
                    <SkillTag key={s} skill={s} variant="missing" index={i} />
                  ))}
                </div>
              </section>
            )}

            {/* Suggestions */}
            {candidate.suggestions?.length > 0 && (
              <section>
                <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
                  Improvement Suggestions
                </h3>
                <div className="space-y-2">
                  {candidate.suggestions.map((s, i) => (
                    <SuggestionCard key={i} text={s} index={i} />
                  ))}
                </div>
              </section>
            )}
          </div>
        </motion.aside>
      </>
    </AnimatePresence>
  );
}
