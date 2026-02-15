-- Add paper_category to levels for grouping in Papers tab (e.g. 'vision', 'language').
-- Only used when section = 'papers'.

alter table public.levels
  add column if not exists paper_category text;

comment on column public.levels.paper_category is 'Category for paper levels: vision, language, etc. Used only when section = papers.';
create index if not exists levels_paper_category_idx on public.levels (paper_category) where paper_category is not null;
