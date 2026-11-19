# Deploying pokerbot

The API runs on **Render** and the front end on **Vercel**. Both deploy from the
GitHub repo and redeploy automatically when you push to `main`.

```
Browser ──> Vercel (static React app)
              │ fetch(VITE_API_URL + "/hands", ...)
              ▼
           Render (FastAPI: loads api/strategies/*.json at startup)
```

## 1. API on Render

1. Push the repo to GitHub (already done if `git push` works).
2. In the Render dashboard: **New → Blueprint**, pick the `poker-bot` repo.
   Render reads `render.yaml` and creates a free web service called `pokerbot-api`.
3. When it asks for `ALLOWED_ORIGINS`, leave it blank for now (you'll fill it in
   after step 2 below). Click **Apply**.
4. Wait for the first deploy (2-3 minutes: `pip install ".[api]"`). Then open
   `https://<your-service>.onrender.com/health`. It should show `{"status":"ok"}`.
   Interactive API docs are at `/docs`.

Copy the service URL, e.g. `https://pokerbot-api.onrender.com`.

## 2. Front end on Vercel

1. In Vercel: **Add New → Project**, import the `poker-bot` repo.
2. Set **Root Directory** to `web`. Vercel detects Vite (build `npm run build`,
   output `dist`); leave those defaults.
3. Under **Environment Variables** add
   `VITE_API_URL` = your Render URL from step 1 (no trailing slash).
4. **Deploy.** Copy the production URL, e.g. `https://pokerbot.vercel.app`.

## 3. Connect them (CORS)

The API only answers browsers coming from origins it knows.

1. Render → `pokerbot-api` → **Environment** → set `ALLOWED_ORIGINS` to your
   Vercel URL, e.g. `https://pokerbot.vercel.app`. Several origins can be
   comma-separated (add `http://localhost:5173` if you also want local dev to
   reach the deployed API).
2. Save; Render restarts the service.
3. Open the Vercel URL and play a hand.

## Notes

- **Cold starts.** Render's free tier sleeps after ~15 minutes without traffic.
  The first visit afterwards takes 30-60 seconds; the site shows a "Waking up
  the server" screen meanwhile. A paid instance removes this.
- **Live solves are slower in production.** The free instance has a fraction of a
  CPU, so a custom-stack solve that takes ~1.5 s locally can take 10-20 s there.
  Whole-number stacks are precomputed and instant.
- **Hands live in memory**, so a restart (including waking from sleep) drops
  unfinished hands; the site just deals a new one.
- **Changing `VITE_API_URL`** requires a Vercel redeploy, because Vite bakes
  environment variables into the build.
- **Vercel preview deployments** get their own URLs; add one to
  `ALLOWED_ORIGINS` if you want a preview to reach the API.
