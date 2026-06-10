import { motion } from 'framer-motion';

export default function SkillTag({ skill, variant = 'matched', index = 0 }) {
  const styles = {
    matched: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20',
    missing: 'bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20',
    neutral: 'bg-slate-100 dark:bg-slate-700/50 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600',
  };

  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.04, duration: 0.2 }}
      className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium ${styles[variant]}`}
    >
      {variant === 'matched' && <span className="mr-1">✓</span>}
      {variant === 'missing' && <span className="mr-1">✗</span>}
      {skill}
    </motion.span>
  );
}
