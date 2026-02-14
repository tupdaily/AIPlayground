/**
 * Seed the `levels` table in Supabase with pre-stored challenge graphs.
 * Run from frontend dir: npm run seed-levels  or  npx tsx scripts/seed-levels.ts
 *
 * Requires in .env.local (or env):
 *   NEXT_PUBLIC_SUPABASE_URL
 *   SUPABASE_SERVICE_ROLE_KEY  (bypasses RLS so we can insert into levels)
 */

import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

function loadEnvLocal() {
  for (const f of [".env.local", ".env"]) {
    const p = resolve(projectRoot, f);
    if (existsSync(p)) {
      const content = readFileSync(p, "utf8");
      for (const line of content.split(/\r?\n/)) {
        const trimmed = line.replace(/#.*$/, "").trim();
        const eq = trimmed.indexOf("=");
        if (eq > 0) {
          const key = trimmed.slice(0, eq).trim();
          let val = trimmed.slice(eq + 1).trim();
          if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'")))
            val = val.slice(1, -1);
          if (key && !process.env[key]) process.env[key] = val;
        }
      }
      break;
    }
  }
}
loadEnvLocal();

import { createClient } from "@supabase/supabase-js";
import type { GraphSchema } from "../src/types/graph";

const LEVEL_1_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    {
      id: "input_1",
      type: "input",
      params: {},
      position: { x: 120, y: 200 },
    },
    {
      id: "output_1",
      type: "output",
      params: {},
      position: { x: 420, y: 200 },
    },
  ],
  edges: [],
  metadata: {
    name: "Level 1: Connect input to output",
    created_at: new Date().toISOString(),
    description: "Connect the Input block to the Output block by adding layers in between (e.g. Flatten, Linear, Activation) to build a feed-forward network.",
  },
};

async function main() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !serviceRoleKey) {
    console.error(
      "Missing env: set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in .env.local)"
    );
    process.exit(1);
  }

  const supabase = createClient(url, serviceRoleKey);

  const { data, error } = await supabase
    .from("levels")
    .upsert(
      {
        level_number: 1,
        name: "Connect input to output",
        description: "Connect the Input block to the Output block by adding layers in between (e.g. Flatten, Linear, Activation) to build a feed-forward network.",
        graph_json: LEVEL_1_GRAPH,
      },
      { onConflict: "level_number" }
    )
    .select("id, level_number, name");

  if (error) {
    console.error("Seed failed:", error.message);
    process.exit(1);
  }

  console.log("Levels seeded successfully:", data);
}

main();
