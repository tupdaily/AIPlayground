-- Playgrounds table: stores each user's saved graphs as JSON
create table if not exists public.playgrounds (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null default 'Untitled Model',
  graph_json jsonb not null default '{"version":"1.0","nodes":[],"edges":[],"metadata":{"name":"Untitled Model","created_at":"2020-01-01T00:00:00.000Z"}}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists playgrounds_user_id_idx on public.playgrounds (user_id);
create index if not exists playgrounds_updated_at_idx on public.playgrounds (updated_at desc);

alter table public.playgrounds enable row level security;

create policy "Users can read own playgrounds"
  on public.playgrounds for select
  using (auth.uid() = user_id);

create policy "Users can insert own playgrounds"
  on public.playgrounds for insert
  with check (auth.uid() = user_id);

create policy "Users can update own playgrounds"
  on public.playgrounds for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users can delete own playgrounds"
  on public.playgrounds for delete
  using (auth.uid() = user_id);

-- Optional: trigger to keep updated_at in sync
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists playgrounds_updated_at on public.playgrounds;
create trigger playgrounds_updated_at
  before update on public.playgrounds
  for each row execute function public.set_updated_at();
