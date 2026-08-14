# HireLens Frontend

React + Vite single-page app for the HireLens resume matcher. Two views:

- **Candidate** — upload one resume PDF and a job description, get the overall
  match, the four-component breakdown, missing skills, and suggestions.
- **Recruiter** — upload up to 50 resumes, get them ranked, then filter the
  batch by score, required skills, and seniority.

## Development

```bash
npm ci
npm run dev      # http://localhost:3000
```

The dev server proxies `/api` to `http://localhost:8000`, so run the backend
alongside it:

```bash
uvicorn backend.main:app --reload --port 8000
```

## Scripts

| Command | Purpose |
|---------|---------|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint over `src/` |

## API origin

`src/services/api.js` calls same-origin `/api` by default — nginx proxies it to
the backend in the container, and Vite proxies it in development. To point the
build at a different origin, set `VITE_API_BASE_URL` at **build** time (Vite
inlines `VITE_*` values, so setting it at runtime has no effect):

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

## Layout

```
src/
├── pages/        Landing, Candidate, Recruiter
├── components/   Score ring, score bars, dropzones, slide-over, skill tags
├── services/     axios client and endpoint wrappers
└── hooks/        useTheme (dark/light with localStorage persistence)
```
