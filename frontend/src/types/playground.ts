import type { GraphSchema } from "./graph";

export interface PlaygroundRow {
  id: string;
  user_id: string;
  name: string;
  graph_json: GraphSchema;
  created_at: string;
  updated_at: string;
}
