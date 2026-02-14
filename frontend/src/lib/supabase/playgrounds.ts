import { createClient } from "@/lib/supabase/client";
import type { GraphSchema } from "@/types/graph";
import type { PlaygroundRow } from "@/types/playground";

export async function listPlaygrounds(): Promise<PlaygroundRow[]> {
  const supabase = createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user) return [];

  const { data, error } = await supabase
    .from("playgrounds")
    .select("*")
    .eq("user_id", user.id)
    .order("updated_at", { ascending: false });
  if (error) return [];
  return (data ?? []) as PlaygroundRow[];
}

export async function getPlayground(id: string): Promise<PlaygroundRow | null> {
  const supabase = createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user) return null;

  const { data, error } = await supabase
    .from("playgrounds")
    .select("*")
    .eq("id", id)
    .eq("user_id", user.id)
    .maybeSingle();
  if (error || !data) return null;
  return data as PlaygroundRow;
}

export async function createPlayground(
  graph: GraphSchema,
  name?: string
): Promise<{ id: string } | null> {
  const supabase = createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user) return null;

  const title = name ?? graph.metadata?.name ?? "Untitled Model";
  const { data, error } = await supabase
    .from("playgrounds")
    .insert({
      user_id: user.id,
      name: title,
      graph_json: graph,
    })
    .select("id")
    .single();
  if (error) return null;
  return { id: (data as { id: string }).id };
}

export async function updatePlayground(
  id: string,
  graph: GraphSchema,
  name?: string
): Promise<boolean> {
  const supabase = createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user) return false;

  const payload: { graph_json: GraphSchema; name?: string } = {
    graph_json: graph,
  };
  if (name !== undefined) payload.name = name;
  else payload.name = graph.metadata?.name ?? "Untitled Model";

  const { error } = await supabase
    .from("playgrounds")
    .update(payload)
    .eq("id", id)
    .eq("user_id", user.id);
  return !error;
}

export async function deletePlayground(id: string): Promise<boolean> {
  const supabase = createClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();
  if (userError || !user) return false;

  const { error } = await supabase
    .from("playgrounds")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);
  return !error;
}
