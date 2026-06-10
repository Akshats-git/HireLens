import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { User, Briefcase, Zap, BarChart3, Target, Sparkles, ArrowRight, CheckCircle2 } from 'lucide-react';

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] },
});

const FEATURES = [
  {
    icon: Target,
    color: 'text-blue-500',
    bg: 'bg-blue-500/10',
    title: 'Precision Matching',
    desc: 'Fine-tuned sentence transformers compute semantic similarity — not just keyword overlap.',
  },
  {
    icon: BarChart3,
    color: 'text-emerald-500',
    bg: 'bg-emerald-500/10',
    title: '4-Dimension Scoring',
    desc: 'Skills (40%), Experience (30%), Education (15%), and Keywords (15%) give a holistic picture.',
  },
  {
    icon: Sparkles,
    color: 'text-violet-500',
    bg: 'bg-violet-500/10',
    title: 'Actionable Insights',
    desc: 'Get specific improvement suggestions tailored to each job — not generic resume tips.',
  },
];

const STEPS = [
  { num: '01', title: 'Upload Resume', desc: 'Drop your PDF — we extract and clean the text automatically.' },
  { num: '02', title: 'Paste Job Description', desc: 'Copy the full JD from any job board.' },
  { num: '03', title: 'Get Your Score', desc: 'Receive a detailed breakdown with skills gap analysis in seconds.' },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="relative overflow-hidden">
      {/* Background blobs */}
      <div className="pointer-events-none select-none" aria-hidden>
        <motion.div
          animate={{ scale: [1, 1.05, 1], opacity: [0.4, 0.6, 0.4] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
          className="absolute -top-32 right-1/4 w-[500px] h-[500px] bg-blue-500/8 dark:bg-blue-500/6 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ scale: [1, 1.08, 1], opacity: [0.3, 0.5, 0.3] }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut', delay: 2 }}
          className="absolute top-2/3 -left-24 w-96 h-96 bg-violet-500/6 dark:bg-violet-500/5 rounded-full blur-3xl"
        />
        {/* Dot grid */}
        <div
          className="absolute inset-0 opacity-40 dark:opacity-20"
          style={{
            backgroundImage: 'radial-gradient(circle, #94a3b8 1px, transparent 1px)',
            backgroundSize: '28px 28px',
          }}
        />
      </div>

      {/* ── Hero ────────────────────────────────────────────────────────────── */}
      <section className="relative max-w-5xl mx-auto px-6 pt-20 pb-16 text-center">
        <motion.div {...fadeUp(0)}>
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-blue-200 dark:border-blue-500/20 bg-blue-50 dark:bg-blue-500/5 text-blue-600 dark:text-blue-400 text-xs font-semibold mb-8 uppercase tracking-widest">
            <Zap size={12} />
            Powered by fine-tuned AI · Open Source
          </div>
        </motion.div>

        <motion.h1
          {...fadeUp(0.1)}
          className="text-5xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.05] mb-6"
        >
          <span className="text-slate-900 dark:text-white">Resume Screening,</span>
          <br />
          <span className="bg-gradient-to-r from-blue-500 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Reimagined with AI
          </span>
        </motion.h1>

        <motion.p
          {...fadeUp(0.2)}
          className="text-lg sm:text-xl text-slate-500 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed mb-12"
        >
          HireLens uses a fine-tuned sentence transformer to instantly score resumes, extract skills,
          and surface actionable gaps — for candidates improving their applications and recruiters
          ranking talent at scale.
        </motion.p>

        {/* CTA Cards */}
        <motion.div
          {...fadeUp(0.3)}
          className="flex flex-col sm:flex-row gap-4 justify-center max-w-2xl mx-auto"
        >
          <button
            onClick={() => navigate('/candidate')}
            className="group flex-1 flex items-center justify-between gap-4 p-5 rounded-2xl bg-blue-500 hover:bg-blue-600 text-white transition-all duration-200 shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:-translate-y-0.5"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center shrink-0">
                <User size={20} />
              </div>
              <div className="text-left">
                <div className="font-bold text-base">I'm a Candidate</div>
                <div className="text-blue-100 text-xs mt-0.5">Score your resume against a job</div>
              </div>
            </div>
            <ArrowRight size={18} className="shrink-0 opacity-70 group-hover:translate-x-1 transition-transform" />
          </button>

          <button
            onClick={() => navigate('/recruiter')}
            className="group flex-1 flex items-center justify-between gap-4 p-5 rounded-2xl border-2 border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600 text-slate-900 dark:text-white transition-all duration-200 hover:shadow-md hover:-translate-y-0.5"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-slate-700 flex items-center justify-center shrink-0">
                <Briefcase size={20} className="text-slate-600 dark:text-slate-300" />
              </div>
              <div className="text-left">
                <div className="font-bold text-base">I'm a Recruiter</div>
                <div className="text-slate-400 text-xs mt-0.5">Rank and filter multiple candidates</div>
              </div>
            </div>
            <ArrowRight size={18} className="shrink-0 opacity-40 group-hover:translate-x-1 transition-transform" />
          </button>
        </motion.div>

        {/* Social proof */}
        <motion.div {...fadeUp(0.4)} className="flex flex-wrap items-center justify-center gap-5 mt-10 text-sm text-slate-400 dark:text-slate-500">
          {['P@1: 96%', 'AUC-ROC: 84%', 'NER F1: 77%', '595 skills taxonomy'].map((stat) => (
            <span key={stat} className="flex items-center gap-1.5">
              <CheckCircle2 size={13} className="text-emerald-500" />
              {stat}
            </span>
          ))}
        </motion.div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-14"
        >
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-3">Why HireLens?</h2>
          <p className="text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
            Built with production ML — not heuristics — for results you can actually trust.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map(({ icon: Icon, color, bg, title, desc }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12, duration: 0.5 }}
              className="p-6 rounded-2xl border border-slate-200 dark:border-slate-700/60 bg-white dark:bg-slate-800/40 hover:border-slate-300 dark:hover:border-slate-600 hover:shadow-md dark:hover:shadow-none transition-all duration-200 group"
            >
              <div className={`w-10 h-10 rounded-xl ${bg} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <Icon size={20} className={color} />
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-20 border-t border-slate-200 dark:border-slate-800">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-3">How it works</h2>
          <p className="text-slate-500 dark:text-slate-400">Three steps to your AI-powered match report.</p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {STEPS.map(({ num, title, desc }, i) => (
            <motion.div
              key={num}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15 }}
              className="relative flex flex-col gap-3"
            >
              <span className="text-5xl font-black text-slate-100 dark:text-slate-800 select-none">{num}</span>
              <div>
                <h3 className="font-semibold text-slate-900 dark:text-white mb-1">{title}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">{desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── CTA footer ───────────────────────────────────────────────────────── */}
      <section className="max-w-3xl mx-auto px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="p-10 rounded-3xl bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 shadow-2xl shadow-blue-500/25"
        >
          <h2 className="text-3xl font-bold text-white mb-3">Ready to find your fit?</h2>
          <p className="text-blue-100 mb-8">Analyze your resume in under 5 seconds — completely free.</p>
          <button
            onClick={() => navigate('/candidate')}
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-xl bg-white text-blue-600 font-semibold text-sm hover:bg-blue-50 transition-colors shadow-lg"
          >
            Analyze My Resume <ArrowRight size={16} />
          </button>
        </motion.div>
      </section>
    </div>
  );
}
