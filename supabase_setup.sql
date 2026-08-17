-- RelayGuard learning mode — Supabase setup.
-- Run this in your Supabase project's SQL editor (Dashboard -> SQL Editor).
--
-- What it creates:
--   1. public.training_samples  - metadata row per uploaded training sample
--   2. storage bucket "relayguard-audio" - holds the WAV files
--   3. DEMO RLS policies (see warning below)
--
-- After running this, set the app's environment:
--   SUPABASE_URL=https://<your-project>.supabase.co
--   SUPABASE_KEY=<service-role or anon key>
-- and restart the server. With both set, uploads land here instead of the
-- local ./user_data fallback.

-- ---------------------------------------------------------------------------
-- 1. metadata table
-- ---------------------------------------------------------------------------
create table if not exists public.training_samples (
  id            uuid primary key default gen_random_uuid(),
  file_path     text not null,              -- object path inside the bucket
  label         text not null check (label in ('normal', 'relay')),
  duration_s    double precision,
  net_speech_s  double precision,
  qc_json       jsonb,                      -- QC stats from upload-time analysis
  notes         text default '',
  created_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- 2. storage bucket (idempotent; alternatively create it in the Dashboard:
--    Storage -> New bucket -> name "relayguard-audio", public = off)
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('relayguard-audio', 'relayguard-audio', false)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- 3. RLS policies — PERMISSIVE, FOR DEMO ONLY.
-- WARNING: these allow anyone holding the anon key to read/write/delete all
-- training data and audio. Tighten for production (per-user rows via
-- auth.uid(), or use the service-role key server-side only and disable the
-- anon policies entirely).
-- ---------------------------------------------------------------------------
alter table public.training_samples enable row level security;

drop policy if exists "demo select training_samples" on public.training_samples;
create policy "demo select training_samples"
  on public.training_samples for select using (true);

drop policy if exists "demo insert training_samples" on public.training_samples;
create policy "demo insert training_samples"
  on public.training_samples for insert with check (true);

drop policy if exists "demo delete training_samples" on public.training_samples;
create policy "demo delete training_samples"
  on public.training_samples for delete using (true);

-- storage object policies for the bucket (same DEMO warning applies)
drop policy if exists "demo read relayguard-audio" on storage.objects;
create policy "demo read relayguard-audio"
  on storage.objects for select
  using (bucket_id = 'relayguard-audio');

drop policy if exists "demo write relayguard-audio" on storage.objects;
create policy "demo write relayguard-audio"
  on storage.objects for insert
  with check (bucket_id = 'relayguard-audio');

drop policy if exists "demo update relayguard-audio" on storage.objects;
create policy "demo update relayguard-audio"
  on storage.objects for update
  using (bucket_id = 'relayguard-audio');

drop policy if exists "demo delete relayguard-audio" on storage.objects;
create policy "demo delete relayguard-audio"
  on storage.objects for delete
  using (bucket_id = 'relayguard-audio');
