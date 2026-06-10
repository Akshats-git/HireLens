import axios from 'axios';

const http = axios.create({
  baseURL: '/api',
  timeout: 90_000, // ML inference can take ~5s; bulk 50 resumes ~30s
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

export const getCandidate = (id) => http.get(`/recruiter/candidate/${id}`);

export const filterCandidates = (filters) =>
  http.post('/recruiter/filter', filters);
