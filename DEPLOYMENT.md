# Deployment

## Local run

1. From the repository root, start the API: `venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload`.
2. In `frontend`, run `npm run dev`.

## Supabase

Create a project, then apply `supabase/migrations/202609220001_predictions.sql` as a migration. The database schema intentionally has no browser access: the API/Edge Function must use its server-side service key.

## Vercel

Import the Git repository in Vercel, set the root directory to `frontend`, set `VITE_API_BASE_URL` to the public API URL, then deploy. Vercel is not connected to this Codex session, so no production deployment was created automatically.

## GitHub

The current checkout points to GitLab and the connected GitHub account has no repository that clearly corresponds to this project. Create or select the intended GitHub repository before pushing; do not replace the current remote blindly.
