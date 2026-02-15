-- Custom datasets: stores metadata for user-uploaded datasets
create table if not exists public.datasets (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text,
  format text not null check (format in ('csv', 'image_folder')),
  gcs_path text not null,
  input_shape integer[] not null,
  num_classes integer not null,
  num_samples integer not null default 0,
  file_size_bytes bigint not null default 0,
  class_names text[],
  label_column text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists datasets_user_id_idx on public.datasets (user_id);
create index if not exists datasets_updated_at_idx on public.datasets (updated_at desc);

alter table public.datasets enable row level security;

create policy "Users can read own datasets"
  on public.datasets for select
  to authenticated
  using (auth.uid() = user_id);

create policy "Users can insert own datasets"
  on public.datasets for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users can update own datasets"
  on public.datasets for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users can delete own datasets"
  on public.datasets for delete
  to authenticated
  using (auth.uid() = user_id);

-- Reuse existing updated_at trigger function from playgrounds migration
drop trigger if exists datasets_updated_at on public.datasets;
create trigger datasets_updated_at
  before update on public.datasets
  for each row execute function public.set_updated_at();
