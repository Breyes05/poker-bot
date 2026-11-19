# pokerbot web

React + TypeScript front end for the pokerbot API (Vite).

```bash
npm install
npm run dev        # http://localhost:5173, talks to VITE_API_URL (default http://localhost:8000)
npm run build      # type-check and build to dist/
npm run lint
```

Start the API first, from the repository root:

```bash
uv run --extra api uvicorn api.main:app --reload
```

`src/api.ts` is the typed client; `src/components/` holds the tables, range charts,
and the solver page. Deployment is described in `../DEPLOY.md`.
