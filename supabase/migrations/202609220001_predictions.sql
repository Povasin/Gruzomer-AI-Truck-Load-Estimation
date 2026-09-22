create table if not exists public.predictions (
  transport_id text primary key check (char_length(transport_id) between 1 and 80),
  load_pct numeric(5,2) not null check (load_pct between 0 and 100),
  cargo_type text not null default 'unknown',
  models_count integer not null default 0 check (models_count >= 0),
  backbone text not null default 'unknown',
  created_at timestamptz not null default now()
);

alter table public.predictions
  add column if not exists cargo_type text not null default 'unknown';
alter table public.predictions
  add column if not exists models_count integer not null default 0;
alter table public.predictions
  add column if not exists backbone text not null default 'unknown';
alter table public.predictions
  add constraint predictions_models_count_nonnegative check (models_count >= 0);

alter table public.predictions enable row level security;

-- The Edge Function uses the service role; browser clients have no direct table access.
revoke all on public.predictions from anon, authenticated;
