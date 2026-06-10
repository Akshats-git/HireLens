function Pulse({ className }) {
  return (
    <div className={`animate-pulse rounded bg-slate-200 dark:bg-slate-700 ${className}`} />
  );
}

export default function ResultsSkeleton() {
  return (
    <div className="space-y-6">
      {/* Score ring placeholder */}
      <div className="flex justify-center">
        <Pulse className="w-36 h-36 rounded-full" />
      </div>

      {/* Score bars */}
      <div className="space-y-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="space-y-2">
            <div className="flex justify-between">
              <Pulse className="h-4 w-32" />
              <Pulse className="h-4 w-10" />
            </div>
            <Pulse className="h-2 w-full" />
          </div>
        ))}
      </div>

      {/* Skills */}
      <div className="space-y-2">
        <Pulse className="h-4 w-28" />
        <div className="flex flex-wrap gap-2">
          {[80, 64, 96, 72, 88].map((w, i) => (
            <Pulse key={i} className={`h-6 w-${w > 80 ? '24' : w > 64 ? '20' : '16'} rounded-md`} style={{ width: w }} />
          ))}
        </div>
      </div>

      {/* Suggestions */}
      <div className="space-y-2">
        <Pulse className="h-4 w-24" />
        {[1, 2, 3].map((i) => (
          <div key={i} className="p-4 rounded-xl border border-slate-200 dark:border-slate-700 space-y-2">
            <Pulse className="h-4 w-3/4" />
            <Pulse className="h-3 w-full" />
            <Pulse className="h-3 w-2/3" />
          </div>
        ))}
      </div>
    </div>
  );
}
