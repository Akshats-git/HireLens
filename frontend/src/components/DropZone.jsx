import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, FileText, X, CheckCircle2 } from 'lucide-react';

// ── Single-file dropzone (Candidate page) ─────────────────────────────────────

export function SingleDropZone({ file, onDrop, onRemove }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
    onDrop: (accepted) => accepted[0] && onDrop(accepted[0]),
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`relative flex flex-col items-center justify-center gap-3 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
          isDragActive
            ? 'border-blue-500 bg-blue-500/5 dark:bg-blue-500/10'
            : file
            ? 'border-emerald-500/50 bg-emerald-500/5 dark:bg-emerald-500/10'
            : 'border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-50 dark:hover:bg-slate-800/50'
        }`}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {file ? (
            <motion.div
              key="file"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex flex-col items-center gap-2 text-center"
            >
              <CheckCircle2 size={32} className="text-emerald-500" />
              <div>
                <p className="font-semibold text-emerald-600 dark:text-emerald-400">{file.name}</p>
                <p className="text-xs text-slate-400 mt-0.5">{(file.size / 1024).toFixed(0)} KB</p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-2 text-center"
            >
              <div className={`p-3 rounded-full transition-colors ${isDragActive ? 'bg-blue-500/20' : 'bg-slate-100 dark:bg-slate-800'}`}>
                <UploadCloud size={24} className={isDragActive ? 'text-blue-500' : 'text-slate-400'} />
              </div>
              <div>
                <p className="font-medium text-slate-700 dark:text-slate-300">
                  {isDragActive ? 'Drop your resume here' : 'Drag & drop your resume'}
                </p>
                <p className="text-xs text-slate-400 mt-0.5">or click to browse — PDF only, max 10 MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {file && (
        <button
          onClick={onRemove}
          className="mt-2 text-xs text-slate-400 hover:text-red-500 transition-colors flex items-center gap-1"
        >
          <X size={12} /> Remove file
        </button>
      )}
    </div>
  );
}

// ── Multi-file dropzone (Recruiter page) ──────────────────────────────────────

export function MultiDropZone({ files, onDrop, onRemove }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true,
    onDrop: (accepted) => onDrop(accepted),
  });

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={`flex flex-col items-center justify-center gap-3 p-6 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
          isDragActive
            ? 'border-blue-500 bg-blue-500/5 dark:bg-blue-500/10'
            : 'border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-slate-50 dark:hover:bg-slate-800/50'
        }`}
      >
        <input {...getInputProps()} />
        <div className={`p-2.5 rounded-full ${isDragActive ? 'bg-blue-500/20' : 'bg-slate-100 dark:bg-slate-800'}`}>
          <UploadCloud size={20} className={isDragActive ? 'text-blue-500' : 'text-slate-400'} />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
            {isDragActive ? 'Drop resumes here' : 'Drop multiple PDFs or click to browse'}
          </p>
          <p className="text-xs text-slate-400 mt-0.5">Up to 50 PDFs · 10 MB each</p>
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
          <AnimatePresence>
            {files.map((file, i) => (
              <motion.div
                key={file.name + i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.15 }}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-800 group"
              >
                <FileText size={14} className="text-blue-500 shrink-0" />
                <span className="text-xs text-slate-700 dark:text-slate-300 truncate flex-1">{file.name}</span>
                <span className="text-xs text-slate-400 shrink-0">{(file.size / 1024).toFixed(0)} KB</span>
                <button
                  onClick={() => onRemove(i)}
                  className="text-slate-300 dark:text-slate-600 hover:text-red-500 transition-colors shrink-0"
                >
                  <X size={12} />
                </button>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
