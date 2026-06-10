import { motion } from 'framer-motion';

const RADIUS = 52;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function getColor(score) {
  if (score >= 80) return '#10b981'; // emerald-500
  if (score >= 60) return '#f59e0b'; // amber-500
  return '#ef4444';                  // red-500
}

export default function CircularScore({ score, size = 160, showLabel = true }) {
  const offset = CIRCUMFERENCE - (score / 100) * CIRCUMFERENCE;
  const color = getColor(score);
  const label = score >= 80 ? 'Excellent' : score >= 70 ? 'Good' : score >= 50 ? 'Fair' : 'Poor';

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox="0 0 120 120">
          {/* Track */}
          <circle
            cx="60" cy="60" r={RADIUS}
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-slate-200 dark:text-slate-700"
          />
          {/* Progress arc */}
          <motion.circle
            cx="60" cy="60" r={RADIUS}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            initial={{ strokeDashoffset: CIRCUMFERENCE }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
            transform="rotate(-90 60 60)"
            style={{ filter: `drop-shadow(0 0 6px ${color}60)` }}
          />
        </svg>

        {/* Centered text overlay */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="text-3xl font-extrabold leading-none"
            style={{ color }}
          >
            {score.toFixed(0)}
          </motion.span>
          <span className="text-xs text-slate-400 dark:text-slate-500 font-medium">/ 100</span>
        </div>
      </div>

      {showLabel && (
        <motion.span
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-sm font-semibold px-3 py-1 rounded-full"
          style={{ color, backgroundColor: `${color}18` }}
        >
          {label}
        </motion.span>
      )}
    </div>
  );
}
