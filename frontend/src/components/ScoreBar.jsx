import { motion } from 'framer-motion';

const LABELS = {
  skills_match:           { label: 'Skills Match',         weight: '40%' },
  experience_relevance:   { label: 'Experience Relevance', weight: '30%' },
  education_fit:          { label: 'Education Fit',        weight: '15%' },
  keyword_alignment:      { label: 'Keyword Alignment',    weight: '15%' },
};

function barColor(value) {
  if (value >= 0.75) return 'from-emerald-500 to-emerald-400';
  if (value >= 0.50) return 'from-amber-500 to-amber-400';
  return 'from-red-500 to-red-400';
}

export default function ScoreBar({ metric, value, delay = 0 }) {
  const { label, weight } = LABELS[metric] || { label: metric, weight: '' };
  const pct = Math.round(value * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-700 dark:text-slate-300">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 dark:text-slate-500">{weight}</span>
          <span className="font-semibold text-slate-900 dark:text-white w-9 text-right">{pct}%</span>
        </div>
      </div>
      <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${barColor(value)}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay }}
        />
      </div>
    </div>
  );
}
