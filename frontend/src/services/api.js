import axios from 'axios';

// Same-origin by default: nginx proxies /api to the backend in both the
// container and the EC2 host config, and Vite proxies it in development.
// Override at build time for split-origin deployments.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const http = axios.create({
  baseURL: BASE_URL,
  // A single resume takes a few seconds; a 50-resume batch can take ~30s.
  timeout: 180_000,
});

// Normalise error messages from FastAPI detail field
http.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const detail = err.response?.data?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join('; ')
      : detail || err.message || 'An unexpected error occurred.';
    return Promise.reject(new Error(message));
  },
);

// ── Candidate ─────────────────────────────────────────────────────────────────

export const analyzeResume = (resumeFile, jobDescription) => {
  const form = new FormData();
  form.append('resume', resumeFile);
  form.append('job_description', jobDescription);
  return http.post('/candidate/analyze', form);
};

// ── Recruiter ─────────────────────────────────────────────────────────────────

export const bulkAnalyze = (resumeFiles, jobDescription) => {
  const form = new FormData();
  resumeFiles.forEach((f) => form.append('resumes', f));
  form.append('job_description', jobDescription);
  return http.post('/recruiter/bulk-analyze', form);
};

export const getCandidate = (batchId, candidateId) =>
  http.get(
    `/recruiter/batches/${encodeURIComponent(batchId)}/candidates/${encodeURIComponent(candidateId)}`,
  );

// Filtering runs server-side against a specific batch, so the batch_id returned
// by bulkAnalyze must be passed back with the filters.
export const filterCandidates = (batchId, filters) =>
  http.post('/recruiter/filter', { batch_id: batchId, ...filters });
