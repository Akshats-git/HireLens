import { motion } from 'framer-motion';
import { Lightbulb } from 'lucide-react';

export default function SuggestionCard({ text, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08, duration: 0.35 }}
      className="flex gap-3 p-4 rounded-xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/5"
    >
      <div className="shrink-0 mt-0.5">
        <div className="w-6 h-6 rounded-full bg-amber-100 dark:bg-amber-500/20 flex items-center justify-center">
          <Lightbulb size={12} className="text-amber-600 dark:text-amber-400" />
        </div>
      </div>
      <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">{text}</p>
    </motion.div>
  );
}
