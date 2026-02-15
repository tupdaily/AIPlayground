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

/** Level 1 correct answer: Input → Flatten → Linear → Output */
const LEVEL_1_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 280, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 128 }, position: { x: 480, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 680, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "linear_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "Level 1 solution",
    created_at: new Date().toISOString(),
  },
};

// Level 2: Add ReLU activation
const LEVEL_2_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 520, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 2",
    created_at: new Date().toISOString(),
    description: "Add a non-linear activation (ReLU) between the Linear layer and the Output.",
  },
};

const LEVEL_2_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 200, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 128 }, position: { x: 320, y: 200 } },
    { id: "activation_1", type: "relu", params: {}, position: { x: 440, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 560, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "linear_1", sourceHandle: "out", target: "activation_1", targetHandle: "in" },
    { id: "e4", source: "activation_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 2 solution", created_at: new Date().toISOString() },
};

// Level 3: Simple CNN (Conv2D → Activation → Flatten → Linear → Output)
const LEVEL_3_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 600, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 3",
    created_at: new Date().toISOString(),
    description: "Build a small convolutional network: Conv2D → Activation → Flatten → Linear → Output.",
  },
};

const LEVEL_3_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv_1", type: "conv2d", params: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 200, y: 200 } },
    { id: "activation_1", type: "relu", params: {}, position: { x: 320, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 440, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 32 * 28 * 28, out_features: 128 }, position: { x: 560, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 680, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "conv_1", targetHandle: "in" },
    { id: "e2", source: "conv_1", sourceHandle: "out", target: "activation_1", targetHandle: "in" },
    { id: "e3", source: "activation_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e4", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e5", source: "linear_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 3 solution", created_at: new Date().toISOString() },
};

// Level 4: Dropout for regularization
const LEVEL_4_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 520, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 4",
    created_at: new Date().toISOString(),
    description: "Add Dropout between Linear and Output to reduce overfitting.",
  },
};

const LEVEL_4_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 200, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 128 }, position: { x: 320, y: 200 } },
    { id: "dropout_1", type: "dropout", params: { p: 0.5 }, position: { x: 440, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 560, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "linear_1", sourceHandle: "out", target: "dropout_1", targetHandle: "in" },
    { id: "e4", source: "dropout_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 4 solution", created_at: new Date().toISOString() },
};

// Level 5: Transformer-style (LayerNorm → Attention → Output). Uses 3D input [B, seq, dim]; Add available for later challenges.
const LEVEL_5_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 480, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 5",
    created_at: new Date().toISOString(),
    description: "Build a transformer-style path: LayerNorm then Attention. (Input must be 3D [batch, seq, features]; e.g. use Embedding first for token sequences.)",
  },
};

const LEVEL_5_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "ln1", type: "layernorm", params: {}, position: { x: 200, y: 200 } },
    { id: "attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 320, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 440, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "ln1", targetHandle: "in" },
    { id: "e2", source: "ln1", sourceHandle: "out", target: "attn", targetHandle: "in" },
    { id: "e3", source: "attn", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 5 solution", created_at: new Date().toISOString() },
};

// Level 6: Residual connection with Add (teaches Add block)
const LEVEL_6_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 520, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 6",
    created_at: new Date().toISOString(),
    description: "Use the Add block with two inputs: a skip path and a path through Linear. Connect both of Add’s input ports.",
  },
};

const LEVEL_6_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 180, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 784 }, position: { x: 300, y: 200 } },
    { id: "add_1", type: "add", params: {}, position: { x: 440, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 560, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "flatten_1", sourceHandle: "out", target: "add_1", targetHandle: "in_a" },
    { id: "e4", source: "linear_1", sourceHandle: "out", target: "add_1", targetHandle: "in_b" },
    { id: "e5", source: "add_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 6 solution", created_at: new Date().toISOString() },
};

// Level 7 (challenges): Two-layer MLP — Flatten → Linear → ReLU → Linear → Output
const LEVEL_7_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 560, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 7: Two-layer MLP",
    created_at: new Date().toISOString(),
    description: "Build a two-layer feedforward network: Flatten → Linear → ReLU → Linear → Output.",
  },
};

const LEVEL_7_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 200, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 256 }, position: { x: 320, y: 200 } },
    { id: "activation_1", type: "relu", params: {}, position: { x: 440, y: 200 } },
    { id: "linear_2", type: "linear", params: { in_features: 256, out_features: 128 }, position: { x: 560, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 680, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "linear_1", sourceHandle: "out", target: "activation_1", targetHandle: "in" },
    { id: "e4", source: "activation_1", sourceHandle: "out", target: "linear_2", targetHandle: "in" },
    { id: "e5", source: "linear_2", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 7 solution", created_at: new Date().toISOString() },
};

// Level 8 (challenges): CNN with MaxPool — Conv2D → ReLU → MaxPool2D → Flatten → Linear → Output
const LEVEL_8_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 600, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 8: CNN with MaxPool",
    created_at: new Date().toISOString(),
    description: "Build a CNN with pooling: Conv2D → ReLU → MaxPool2D → Flatten → Linear → Output.",
  },
};

const LEVEL_8_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv_1", type: "conv2d", params: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 200, y: 200 } },
    { id: "activation_1", type: "relu", params: {}, position: { x: 320, y: 200 } },
    { id: "pool_1", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 440, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 560, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 32 * 14 * 14, out_features: 128 }, position: { x: 680, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 800, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "conv_1", targetHandle: "in" },
    { id: "e2", source: "conv_1", sourceHandle: "out", target: "activation_1", targetHandle: "in" },
    { id: "e3", source: "activation_1", sourceHandle: "out", target: "pool_1", targetHandle: "in" },
    { id: "e4", source: "pool_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e5", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e6", source: "linear_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 8 solution", created_at: new Date().toISOString() },
};

// Level 9 (challenges): BatchNorm — Flatten → Linear → BatchNorm → ReLU → Output
const LEVEL_9_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 520, y: 200 } },
  ],
  edges: [],
  metadata: {
    name: "Level 9: BatchNorm",
    created_at: new Date().toISOString(),
    description: "Add BatchNorm between Linear and Output: Flatten → Linear → BatchNorm → ReLU → Output.",
  },
};

const LEVEL_9_SOLUTION: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 200, y: 200 } },
    { id: "linear_1", type: "linear", params: { in_features: 784, out_features: 128 }, position: { x: 320, y: 200 } },
    { id: "bn_1", type: "batchnorm", params: {}, position: { x: 440, y: 200 } },
    { id: "activation_1", type: "relu", params: {}, position: { x: 560, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 680, y: 200 } },
  ],
  edges: [
    { id: "e1", source: "input_1", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e2", source: "flatten_1", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e3", source: "linear_1", sourceHandle: "out", target: "bn_1", targetHandle: "in" },
    { id: "e4", source: "bn_1", sourceHandle: "out", target: "activation_1", targetHandle: "in" },
    { id: "e5", source: "activation_1", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: { name: "Level 9 solution", created_at: new Date().toISOString() },
};

// ---------------------------------------------------------------------------
// Paper: Attention is All You Need (Vaswani et al.) — pre-built with text pipeline
// ---------------------------------------------------------------------------

/** Encoder layout: elevated Add blocks so skip connections are visible. Y increases downward. */
const TRANSFORMER_SKIP_Y = 220; // elevated y for add_1 (skip from pos_embed visible)
const TRANSFORMER_SKIP_Y2 = 120; // elevated y for add_2 (skip from ln_mid visible)
const TRANSFORMER_MAIN_Y = 300; // main path (attn, FFN) slightly lower
const TRANSFORMER_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    // Bottom: Inputs → Input Embedding → Positional Encoding
    { id: "text_input_1", type: "text_input", params: { batch_size: 1, seq_len: 128 }, position: { x: 80, y: 380 } },
    { id: "text_embed_1", type: "text_embedding", params: { vocab_size: 10000, embedding_dim: 128 }, position: { x: 220, y: 380 } },
    { id: "pos_embed_1", type: "positional_embedding", params: { d_model: 128, max_len: 512 }, position: { x: 360, y: 380 } },
    // Encoder layer 1: Attention sublayer — Add elevated for visible skip
    { id: "ln_pre", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: TRANSFORMER_MAIN_Y } },
    { id: "attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 460, y: TRANSFORMER_MAIN_Y } },
    { id: "add_1", type: "add", params: {}, position: { x: 600, y: TRANSFORMER_SKIP_Y } },
    // Encoder layer 2: FFN sublayer — Add elevated for visible skip
    { id: "ln_mid", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: TRANSFORMER_MAIN_Y } },
    { id: "linear_1", type: "linear", params: { in_features: 128, out_features: 512 }, position: { x: 460, y: TRANSFORMER_MAIN_Y } },
    { id: "relu_1", type: "activation", params: { activation: "relu" }, position: { x: 540, y: TRANSFORMER_MAIN_Y } },
    { id: "linear_2", type: "linear", params: { in_features: 512, out_features: 128 }, position: { x: 620, y: TRANSFORMER_MAIN_Y } },
    { id: "add_2", type: "add", params: {}, position: { x: 760, y: TRANSFORMER_SKIP_Y2 } },
    // Top: final LayerNorm → Output
    { id: "ln_post", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 60 } },
    { id: "output_1", type: "output", params: {}, position: { x: 460, y: 60 } },
  ],
  edges: [
    { id: "e0a", source: "text_input_1", sourceHandle: "out", target: "text_embed_1", targetHandle: "in" },
    { id: "e0b", source: "text_embed_1", sourceHandle: "out", target: "pos_embed_1", targetHandle: "in" },
    { id: "e1", source: "pos_embed_1", sourceHandle: "out", target: "ln_pre", targetHandle: "in" },
    { id: "e2", source: "pos_embed_1", sourceHandle: "out", target: "add_1", targetHandle: "in_a" },
    { id: "e3", source: "ln_pre", sourceHandle: "out", target: "attn", targetHandle: "in" },
    { id: "e4", source: "attn", sourceHandle: "out", target: "add_1", targetHandle: "in_b" },
    { id: "e5", source: "add_1", sourceHandle: "out", target: "ln_mid", targetHandle: "in" },
    { id: "e6", source: "ln_mid", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e7", source: "ln_mid", sourceHandle: "out", target: "add_2", targetHandle: "in_a" },
    { id: "e8", source: "linear_1", sourceHandle: "out", target: "relu_1", targetHandle: "in" },
    { id: "e9", source: "relu_1", sourceHandle: "out", target: "linear_2", targetHandle: "in" },
    { id: "e10", source: "linear_2", sourceHandle: "out", target: "add_2", targetHandle: "in_b" },
    { id: "e11", source: "add_2", sourceHandle: "out", target: "ln_post", targetHandle: "in" },
    { id: "e12", source: "ln_post", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "Transformer (Vaswani et al.)",
    created_at: new Date().toISOString(),
    description: "Pre-built Transformer with text pipeline: Text Input → Text Embedding → Positional Embedding → one Encoder block (Self-Attention + FFN with residuals). Explore, run, and modify. Shapes: [B, seq] → [B, seq, 128] → encoder.",
  },
};

// ---------------------------------------------------------------------------
// Paper: ImageNet Classification with Deep CNNs (Krizhevsky et al. — AlexNet)
// ---------------------------------------------------------------------------

const ALEXNET_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv1", type: "conv2d", params: { in_channels: 1, out_channels: 96, kernel_size: 11, stride: 4, padding: 2 }, position: { x: 200, y: 200 } },
    { id: "relu1", type: "activation", params: { activation: "relu" }, position: { x: 320, y: 200 } },
    { id: "conv2", type: "conv2d", params: { in_channels: 96, out_channels: 256, kernel_size: 5, stride: 1, padding: 2 }, position: { x: 440, y: 200 } },
    { id: "relu2", type: "activation", params: { activation: "relu" }, position: { x: 560, y: 200 } },
    { id: "conv3", type: "conv2d", params: { in_channels: 256, out_channels: 384, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 680, y: 200 } },
    { id: "relu3", type: "activation", params: { activation: "relu" }, position: { x: 800, y: 200 } },
    { id: "conv4", type: "conv2d", params: { in_channels: 384, out_channels: 384, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 920, y: 200 } },
    { id: "relu4", type: "activation", params: { activation: "relu" }, position: { x: 1040, y: 200 } },
    { id: "conv5", type: "conv2d", params: { in_channels: 384, out_channels: 256, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 1160, y: 200 } },
    { id: "relu5", type: "activation", params: { activation: "relu" }, position: { x: 1280, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 1400, y: 200 } },
    { id: "fc6", type: "linear", params: { in_features: 9216, out_features: 4096 }, position: { x: 1520, y: 200 } },
    { id: "relu6", type: "activation", params: { activation: "relu" }, position: { x: 1640, y: 200 } },
    { id: "dropout1", type: "dropout", params: { p: 0.5 }, position: { x: 1760, y: 200 } },
    { id: "fc7", type: "linear", params: { in_features: 4096, out_features: 4096 }, position: { x: 1880, y: 200 } },
    { id: "relu7", type: "activation", params: { activation: "relu" }, position: { x: 2000, y: 200 } },
    { id: "dropout2", type: "dropout", params: { p: 0.5 }, position: { x: 2120, y: 200 } },
    { id: "fc8", type: "linear", params: { in_features: 4096, out_features: 1000 }, position: { x: 2240, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 2360, y: 200 } },
  ],
  edges: [
    { id: "e0", source: "input_1", sourceHandle: "out", target: "conv1", targetHandle: "in" },
    { id: "e1", source: "conv1", sourceHandle: "out", target: "relu1", targetHandle: "in" },
    { id: "e2", source: "relu1", sourceHandle: "out", target: "conv2", targetHandle: "in" },
    { id: "e3", source: "conv2", sourceHandle: "out", target: "relu2", targetHandle: "in" },
    { id: "e4", source: "relu2", sourceHandle: "out", target: "conv3", targetHandle: "in" },
    { id: "e5", source: "conv3", sourceHandle: "out", target: "relu3", targetHandle: "in" },
    { id: "e6", source: "relu3", sourceHandle: "out", target: "conv4", targetHandle: "in" },
    { id: "e7", source: "conv4", sourceHandle: "out", target: "relu4", targetHandle: "in" },
    { id: "e8", source: "relu4", sourceHandle: "out", target: "conv5", targetHandle: "in" },
    { id: "e9", source: "conv5", sourceHandle: "out", target: "relu5", targetHandle: "in" },
    { id: "e10", source: "relu5", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e11", source: "flatten_1", sourceHandle: "out", target: "fc6", targetHandle: "in" },
    { id: "e12", source: "fc6", sourceHandle: "out", target: "relu6", targetHandle: "in" },
    { id: "e13", source: "relu6", sourceHandle: "out", target: "dropout1", targetHandle: "in" },
    { id: "e14", source: "dropout1", sourceHandle: "out", target: "fc7", targetHandle: "in" },
    { id: "e15", source: "fc7", sourceHandle: "out", target: "relu7", targetHandle: "in" },
    { id: "e16", source: "relu7", sourceHandle: "out", target: "dropout2", targetHandle: "in" },
    { id: "e17", source: "dropout2", sourceHandle: "out", target: "fc8", targetHandle: "in" },
    { id: "e18", source: "fc8", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "AlexNet (Krizhevsky et al.)",
    created_at: new Date().toISOString(),
    description: "AlexNet architecture: 5 conv layers (96→256→384→384→256 filters) with ReLU, Flatten, then 3 FC layers (4096→4096→1000) with dropout. Built for ImageNet 224×224×3 input.",
  },
};

// ---------------------------------------------------------------------------
// Paper: Deep Residual Learning for Image Recognition (He et al. — ResNet)
// ---------------------------------------------------------------------------

const RESNET_SKIP_Y = 100; // elevated y for Add blocks so skip wires are visible
const RESNET_BRANCH_Y = 280; // lower y for residual branch (conv→bn→relu→conv→bn)
const RESNET_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv1", type: "conv2d", params: { in_channels: 1, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 200, y: 200 } },
    { id: "bn1", type: "batchnorm", params: {}, position: { x: 320, y: 200 } },
    { id: "relu1", type: "activation", params: { activation: "relu" }, position: { x: 440, y: 200 } },
    // Block 1: residual branch lowered; add1 elevated for visible skip
    { id: "conv2a", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 560, y: RESNET_BRANCH_Y } },
    { id: "bn2a", type: "batchnorm", params: {}, position: { x: 680, y: RESNET_BRANCH_Y } },
    { id: "relu2a", type: "activation", params: { activation: "relu" }, position: { x: 800, y: RESNET_BRANCH_Y } },
    { id: "conv2b", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 920, y: RESNET_BRANCH_Y } },
    { id: "bn2b", type: "batchnorm", params: {}, position: { x: 1040, y: RESNET_BRANCH_Y } },
    { id: "add1", type: "add", params: {}, position: { x: 1160, y: RESNET_SKIP_Y } },
    { id: "relu2b", type: "activation", params: { activation: "relu" }, position: { x: 1280, y: RESNET_SKIP_Y } },
    // Block 2: same pattern
    { id: "conv3a", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 1400, y: RESNET_BRANCH_Y } },
    { id: "bn3a", type: "batchnorm", params: {}, position: { x: 1520, y: RESNET_BRANCH_Y } },
    { id: "relu3a", type: "activation", params: { activation: "relu" }, position: { x: 1640, y: RESNET_BRANCH_Y } },
    { id: "conv3b", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 1760, y: RESNET_BRANCH_Y } },
    { id: "bn3b", type: "batchnorm", params: {}, position: { x: 1880, y: RESNET_BRANCH_Y } },
    { id: "add2", type: "add", params: {}, position: { x: 2000, y: RESNET_SKIP_Y } },
    { id: "relu3b", type: "activation", params: { activation: "relu" }, position: { x: 2120, y: RESNET_SKIP_Y } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 2240, y: RESNET_SKIP_Y } },
    { id: "fc", type: "linear", params: { in_features: 50176, out_features: 10 }, position: { x: 2360, y: RESNET_SKIP_Y } },
    { id: "output_1", type: "output", params: {}, position: { x: 2480, y: RESNET_SKIP_Y } },
  ],
  edges: [
    { id: "e0", source: "input_1", sourceHandle: "out", target: "conv1", targetHandle: "in" },
    { id: "e1", source: "conv1", sourceHandle: "out", target: "bn1", targetHandle: "in" },
    { id: "e2", source: "bn1", sourceHandle: "out", target: "relu1", targetHandle: "in" },
    { id: "e3", source: "relu1", sourceHandle: "out", target: "conv2a", targetHandle: "in" },
    { id: "e4", source: "conv2a", sourceHandle: "out", target: "bn2a", targetHandle: "in" },
    { id: "e5", source: "bn2a", sourceHandle: "out", target: "relu2a", targetHandle: "in" },
    { id: "e6", source: "relu2a", sourceHandle: "out", target: "conv2b", targetHandle: "in" },
    { id: "e7", source: "conv2b", sourceHandle: "out", target: "bn2b", targetHandle: "in" },
    { id: "e8a", source: "relu1", sourceHandle: "out", target: "add1", targetHandle: "in_a" },
    { id: "e8b", source: "bn2b", sourceHandle: "out", target: "add1", targetHandle: "in_b" },
    { id: "e9", source: "add1", sourceHandle: "out", target: "relu2b", targetHandle: "in" },
    { id: "e10", source: "relu2b", sourceHandle: "out", target: "conv3a", targetHandle: "in" },
    { id: "e11", source: "conv3a", sourceHandle: "out", target: "bn3a", targetHandle: "in" },
    { id: "e12", source: "bn3a", sourceHandle: "out", target: "relu3a", targetHandle: "in" },
    { id: "e13", source: "relu3a", sourceHandle: "out", target: "conv3b", targetHandle: "in" },
    { id: "e14", source: "conv3b", sourceHandle: "out", target: "bn3b", targetHandle: "in" },
    { id: "e15a", source: "relu2b", sourceHandle: "out", target: "add2", targetHandle: "in_a" },
    { id: "e15b", source: "bn3b", sourceHandle: "out", target: "add2", targetHandle: "in_b" },
    { id: "e16", source: "add2", sourceHandle: "out", target: "relu3b", targetHandle: "in" },
    { id: "e17", source: "relu3b", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e18", source: "flatten_1", sourceHandle: "out", target: "fc", targetHandle: "in" },
    { id: "e19", source: "fc", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "ResNet (He et al.)",
    created_at: new Date().toISOString(),
    description: "Simplified ResNet: stem (Conv→BN→ReLU) plus two residual blocks. Each block: Conv→BN→ReLU→Conv→BN, then Add(identity, output). Flatten and linear to 10 classes. For 28×28 input.",
  },
};

// ---------------------------------------------------------------------------
// Paper: BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.)
// ---------------------------------------------------------------------------

const BERT_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "text_input_1", type: "text_input", params: { batch_size: 1, seq_len: 128 }, position: { x: 80, y: 380 } },
    { id: "text_embed_1", type: "text_embedding", params: { vocab_size: 30522, embedding_dim: 128 }, position: { x: 220, y: 380 } },
    { id: "pos_embed_1", type: "positional_embedding", params: { d_model: 128, max_len: 512 }, position: { x: 360, y: 380 } },
    { id: "ln_pre", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 300 } },
    { id: "attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 460, y: 300 } },
    { id: "add_1", type: "add", params: {}, position: { x: 600, y: 220 } },
    { id: "ln_mid", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 300 } },
    { id: "linear_1", type: "linear", params: { in_features: 128, out_features: 512 }, position: { x: 460, y: 300 } },
    { id: "gelu_1", type: "activation", params: { activation: "gelu" }, position: { x: 540, y: 300 } },
    { id: "linear_2", type: "linear", params: { in_features: 512, out_features: 128 }, position: { x: 620, y: 300 } },
    { id: "add_2", type: "add", params: {}, position: { x: 760, y: 120 } },
    { id: "ln_post", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 60 } },
    { id: "output_1", type: "output", params: {}, position: { x: 460, y: 60 } },
  ],
  edges: [
    { id: "e0a", source: "text_input_1", sourceHandle: "out", target: "text_embed_1", targetHandle: "in" },
    { id: "e0b", source: "text_embed_1", sourceHandle: "out", target: "pos_embed_1", targetHandle: "in" },
    { id: "e1", source: "pos_embed_1", sourceHandle: "out", target: "ln_pre", targetHandle: "in" },
    { id: "e2", source: "pos_embed_1", sourceHandle: "out", target: "add_1", targetHandle: "in_a" },
    { id: "e3", source: "ln_pre", sourceHandle: "out", target: "attn", targetHandle: "in" },
    { id: "e4", source: "attn", sourceHandle: "out", target: "add_1", targetHandle: "in_b" },
    { id: "e5", source: "add_1", sourceHandle: "out", target: "ln_mid", targetHandle: "in" },
    { id: "e6", source: "ln_mid", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e7", source: "ln_mid", sourceHandle: "out", target: "add_2", targetHandle: "in_a" },
    { id: "e8", source: "linear_1", sourceHandle: "out", target: "gelu_1", targetHandle: "in" },
    { id: "e9", source: "gelu_1", sourceHandle: "out", target: "linear_2", targetHandle: "in" },
    { id: "e10", source: "linear_2", sourceHandle: "out", target: "add_2", targetHandle: "in_b" },
    { id: "e11", source: "add_2", sourceHandle: "out", target: "ln_post", targetHandle: "in" },
    { id: "e12", source: "ln_post", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "BERT (Devlin et al.)",
    created_at: new Date().toISOString(),
    description: "BERT encoder: Text Input → Token Embedding → Position Embedding → one encoder block (Self-Attention + FFN with GELU and residuals). BERT-base has 12 layers, d_model=768; here 1 layer, 128 dim.",
  },
};

// ---------------------------------------------------------------------------
// Paper: Improving Language Understanding by Generative Pre-Training (GPT)
// ---------------------------------------------------------------------------

const GPT_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "text_input_1", type: "text_input", params: { batch_size: 1, seq_len: 128 }, position: { x: 80, y: 380 } },
    { id: "text_embed_1", type: "text_embedding", params: { vocab_size: 50257, embedding_dim: 128 }, position: { x: 220, y: 380 } },
    { id: "pos_embed_1", type: "positional_embedding", params: { d_model: 128, max_len: 1024 }, position: { x: 360, y: 380 } },
    { id: "ln_pre", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 300 } },
    { id: "attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 460, y: 300 } },
    { id: "add_1", type: "add", params: {}, position: { x: 600, y: 220 } },
    { id: "ln_mid", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 300 } },
    { id: "linear_1", type: "linear", params: { in_features: 128, out_features: 512 }, position: { x: 460, y: 300 } },
    { id: "gelu_1", type: "activation", params: { activation: "gelu" }, position: { x: 540, y: 300 } },
    { id: "linear_2", type: "linear", params: { in_features: 512, out_features: 128 }, position: { x: 620, y: 300 } },
    { id: "add_2", type: "add", params: {}, position: { x: 760, y: 120 } },
    { id: "ln_post", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 320, y: 60 } },
    { id: "output_1", type: "output", params: {}, position: { x: 460, y: 60 } },
  ],
  edges: [
    { id: "e0a", source: "text_input_1", sourceHandle: "out", target: "text_embed_1", targetHandle: "in" },
    { id: "e0b", source: "text_embed_1", sourceHandle: "out", target: "pos_embed_1", targetHandle: "in" },
    { id: "e1", source: "pos_embed_1", sourceHandle: "out", target: "ln_pre", targetHandle: "in" },
    { id: "e2", source: "pos_embed_1", sourceHandle: "out", target: "add_1", targetHandle: "in_a" },
    { id: "e3", source: "ln_pre", sourceHandle: "out", target: "attn", targetHandle: "in" },
    { id: "e4", source: "attn", sourceHandle: "out", target: "add_1", targetHandle: "in_b" },
    { id: "e5", source: "add_1", sourceHandle: "out", target: "ln_mid", targetHandle: "in" },
    { id: "e6", source: "ln_mid", sourceHandle: "out", target: "linear_1", targetHandle: "in" },
    { id: "e7", source: "ln_mid", sourceHandle: "out", target: "add_2", targetHandle: "in_a" },
    { id: "e8", source: "linear_1", sourceHandle: "out", target: "gelu_1", targetHandle: "in" },
    { id: "e9", source: "gelu_1", sourceHandle: "out", target: "linear_2", targetHandle: "in" },
    { id: "e10", source: "linear_2", sourceHandle: "out", target: "add_2", targetHandle: "in_b" },
    { id: "e11", source: "add_2", sourceHandle: "out", target: "ln_post", targetHandle: "in" },
    { id: "e12", source: "ln_post", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "GPT (Radford et al.)",
    created_at: new Date().toISOString(),
    description: "GPT decoder: Text Input → Token Embedding → Position Embedding → one decoder block (Masked Self-Attention + FFN with GELU and residuals). Causal masking enables autoregressive next-token prediction. GPT-2 has 12 layers, d_model=768; here 1 layer, 128 dim.",
  },
};

// ---------------------------------------------------------------------------
// Paper: LeNet-5 (LeCun et al.)
// ---------------------------------------------------------------------------

// Block width is ~260px; use 320px spacing so blocks don't overlap.
const LENET5_SPACING = 320;
const LENET5_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv1", type: "conv2d", params: { in_channels: 1, out_channels: 6, kernel_size: 5, stride: 1, padding: 0 }, position: { x: 80 + LENET5_SPACING * 1, y: 200 } },
    { id: "tanh1", type: "activation", params: { activation: "tanh" }, position: { x: 80 + LENET5_SPACING * 2, y: 200 } },
    { id: "pool1", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + LENET5_SPACING * 3, y: 200 } },
    { id: "conv2", type: "conv2d", params: { in_channels: 6, out_channels: 16, kernel_size: 5, stride: 1, padding: 0 }, position: { x: 80 + LENET5_SPACING * 4, y: 200 } },
    { id: "tanh2", type: "activation", params: { activation: "tanh" }, position: { x: 80 + LENET5_SPACING * 5, y: 200 } },
    { id: "pool2", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + LENET5_SPACING * 6, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 80 + LENET5_SPACING * 7, y: 200 } },
    { id: "fc1", type: "linear", params: { in_features: 256, out_features: 120 }, position: { x: 80 + LENET5_SPACING * 8, y: 200 } },
    { id: "tanh3", type: "activation", params: { activation: "tanh" }, position: { x: 80 + LENET5_SPACING * 9, y: 200 } },
    { id: "fc2", type: "linear", params: { in_features: 120, out_features: 84 }, position: { x: 80 + LENET5_SPACING * 10, y: 200 } },
    { id: "tanh4", type: "activation", params: { activation: "tanh" }, position: { x: 80 + LENET5_SPACING * 11, y: 200 } },
    { id: "fc3", type: "linear", params: { in_features: 84, out_features: 10 }, position: { x: 80 + LENET5_SPACING * 12, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 80 + LENET5_SPACING * 13, y: 200 } },
  ],
  edges: [
    { id: "e0", source: "input_1", sourceHandle: "out", target: "conv1", targetHandle: "in" },
    { id: "e1", source: "conv1", sourceHandle: "out", target: "tanh1", targetHandle: "in" },
    { id: "e2", source: "tanh1", sourceHandle: "out", target: "pool1", targetHandle: "in" },
    { id: "e3", source: "pool1", sourceHandle: "out", target: "conv2", targetHandle: "in" },
    { id: "e4", source: "conv2", sourceHandle: "out", target: "tanh2", targetHandle: "in" },
    { id: "e5", source: "tanh2", sourceHandle: "out", target: "pool2", targetHandle: "in" },
    { id: "e6", source: "pool2", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e7", source: "flatten_1", sourceHandle: "out", target: "fc1", targetHandle: "in" },
    { id: "e8", source: "fc1", sourceHandle: "out", target: "tanh3", targetHandle: "in" },
    { id: "e9", source: "tanh3", sourceHandle: "out", target: "fc2", targetHandle: "in" },
    { id: "e10", source: "fc2", sourceHandle: "out", target: "tanh4", targetHandle: "in" },
    { id: "e11", source: "tanh4", sourceHandle: "out", target: "fc3", targetHandle: "in" },
    { id: "e12", source: "fc3", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "LeNet-5 (LeCun et al.)",
    created_at: new Date().toISOString(),
    description: "LeNet-5: Conv(6, 5×5) → tanh → MaxPool(2×2) → Conv(16, 5×5) → tanh → MaxPool(2×2) → Flatten → FC(120) → tanh → FC(84) → tanh → FC(10). Classic architecture for digit recognition; original paper used tanh/sigmoid.",
  },
};

// ---------------------------------------------------------------------------
// Paper: VGG — Very Deep Convolutional Networks (Simonyan & Zisserman)
// ---------------------------------------------------------------------------

// Block width is ~260px; use 320px spacing so blocks don't overlap.
const VGG_SPACING = 320;
const VGG_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 200 } },
    { id: "conv1", type: "conv2d", params: { in_channels: 1, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + VGG_SPACING * 1, y: 200 } },
    { id: "relu1", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 2, y: 200 } },
    { id: "conv2", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + VGG_SPACING * 3, y: 200 } },
    { id: "relu2", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 4, y: 200 } },
    { id: "pool1", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + VGG_SPACING * 5, y: 200 } },
    { id: "conv3", type: "conv2d", params: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + VGG_SPACING * 6, y: 200 } },
    { id: "relu3", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 7, y: 200 } },
    { id: "conv4", type: "conv2d", params: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + VGG_SPACING * 8, y: 200 } },
    { id: "relu4", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 9, y: 200 } },
    { id: "pool2", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + VGG_SPACING * 10, y: 200 } },
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 80 + VGG_SPACING * 11, y: 200 } },
    { id: "fc1", type: "linear", params: { in_features: 128 * 7 * 7, out_features: 4096 }, position: { x: 80 + VGG_SPACING * 12, y: 200 } },
    { id: "relu5", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 13, y: 200 } },
    { id: "dropout1", type: "dropout", params: { p: 0.5 }, position: { x: 80 + VGG_SPACING * 14, y: 200 } },
    { id: "fc2", type: "linear", params: { in_features: 4096, out_features: 4096 }, position: { x: 80 + VGG_SPACING * 15, y: 200 } },
    { id: "relu6", type: "activation", params: { activation: "relu" }, position: { x: 80 + VGG_SPACING * 16, y: 200 } },
    { id: "dropout2", type: "dropout", params: { p: 0.5 }, position: { x: 80 + VGG_SPACING * 17, y: 200 } },
    { id: "fc3", type: "linear", params: { in_features: 4096, out_features: 1000 }, position: { x: 80 + VGG_SPACING * 18, y: 200 } },
    { id: "output_1", type: "output", params: {}, position: { x: 80 + VGG_SPACING * 19, y: 200 } },
  ],
  edges: [
    { id: "e0", source: "input_1", sourceHandle: "out", target: "conv1", targetHandle: "in" },
    { id: "e1", source: "conv1", sourceHandle: "out", target: "relu1", targetHandle: "in" },
    { id: "e2", source: "relu1", sourceHandle: "out", target: "conv2", targetHandle: "in" },
    { id: "e3", source: "conv2", sourceHandle: "out", target: "relu2", targetHandle: "in" },
    { id: "e4", source: "relu2", sourceHandle: "out", target: "pool1", targetHandle: "in" },
    { id: "e5", source: "pool1", sourceHandle: "out", target: "conv3", targetHandle: "in" },
    { id: "e6", source: "conv3", sourceHandle: "out", target: "relu3", targetHandle: "in" },
    { id: "e7", source: "relu3", sourceHandle: "out", target: "conv4", targetHandle: "in" },
    { id: "e8", source: "conv4", sourceHandle: "out", target: "relu4", targetHandle: "in" },
    { id: "e9", source: "relu4", sourceHandle: "out", target: "pool2", targetHandle: "in" },
    { id: "e10", source: "pool2", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e11", source: "flatten_1", sourceHandle: "out", target: "fc1", targetHandle: "in" },
    { id: "e12", source: "fc1", sourceHandle: "out", target: "relu5", targetHandle: "in" },
    { id: "e13", source: "relu5", sourceHandle: "out", target: "dropout1", targetHandle: "in" },
    { id: "e14", source: "dropout1", sourceHandle: "out", target: "fc2", targetHandle: "in" },
    { id: "e15", source: "fc2", sourceHandle: "out", target: "relu6", targetHandle: "in" },
    { id: "e16", source: "relu6", sourceHandle: "out", target: "dropout2", targetHandle: "in" },
    { id: "e17", source: "dropout2", sourceHandle: "out", target: "fc3", targetHandle: "in" },
    { id: "e18", source: "fc3", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "VGG (Simonyan & Zisserman)",
    created_at: new Date().toISOString(),
    description: "VGG-style stack: two blocks of 2×Conv(3×3)+ReLU+MaxPool(2×2), then Flatten and three FC layers (4096→4096→1000) with dropout. All convs use 3×3 filters.",
  },
};

// ---------------------------------------------------------------------------
// Paper: Going Deeper with Convolutions (Szegedy et al. — GoogLeNet / Inception)
// ---------------------------------------------------------------------------

const INCEPTION_SPACING = 280;
const INCEPTION_BRANCH_Y = 120;
/** Inception module: four parallel branches (1×1, 3×3, 5×5, pool+1×1) concatenated along channels. Stem: Conv→ReLU→MaxPool. */
const INCEPTION_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    { id: "input_1", type: "input", params: {}, position: { x: 80, y: 280 } },
    // Stem
    { id: "conv_stem", type: "conv2d", params: { in_channels: 1, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + INCEPTION_SPACING * 1, y: 280 } },
    { id: "relu_stem", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 2, y: 280 } },
    { id: "pool_stem", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + INCEPTION_SPACING * 3, y: 280 } },
    // Branch 1: 1×1 only → pool
    { id: "conv_1x1_a", type: "conv2d", params: { in_channels: 64, out_channels: 64, kernel_size: 1, stride: 1, padding: 0 }, position: { x: 80 + INCEPTION_SPACING * 4, y: 80 } },
    { id: "relu_a", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 5, y: 80 } },
    { id: "pool_a", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + INCEPTION_SPACING * 6, y: 80 } },
    // Branch 2: 1×1 → 3×3 → pool
    { id: "conv_1x1_b", type: "conv2d", params: { in_channels: 64, out_channels: 96, kernel_size: 1, stride: 1, padding: 0 }, position: { x: 80 + INCEPTION_SPACING * 4, y: 80 + INCEPTION_BRANCH_Y } },
    { id: "relu_b1", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 5, y: 80 + INCEPTION_BRANCH_Y } },
    { id: "conv_3x3_b", type: "conv2d", params: { in_channels: 96, out_channels: 128, kernel_size: 3, stride: 1, padding: 1 }, position: { x: 80 + INCEPTION_SPACING * 6, y: 80 + INCEPTION_BRANCH_Y } },
    { id: "relu_b2", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 7, y: 80 + INCEPTION_BRANCH_Y } },
    { id: "pool_b2", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + INCEPTION_SPACING * 8, y: 80 + INCEPTION_BRANCH_Y } },
    // Branch 3: 1×1 → 5×5 → pool
    { id: "conv_1x1_c", type: "conv2d", params: { in_channels: 64, out_channels: 16, kernel_size: 1, stride: 1, padding: 0 }, position: { x: 80 + INCEPTION_SPACING * 4, y: 80 + INCEPTION_BRANCH_Y * 2 } },
    { id: "relu_c1", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 5, y: 80 + INCEPTION_BRANCH_Y * 2 } },
    { id: "conv_5x5_c", type: "conv2d", params: { in_channels: 16, out_channels: 32, kernel_size: 5, stride: 1, padding: 2 }, position: { x: 80 + INCEPTION_SPACING * 6, y: 80 + INCEPTION_BRANCH_Y * 2 } },
    { id: "relu_c2", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 7, y: 80 + INCEPTION_BRANCH_Y * 2 } },
    { id: "pool_c", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + INCEPTION_SPACING * 8, y: 80 + INCEPTION_BRANCH_Y * 2 } },
    // Branch 4: 2×2 maxpool → 1×1
    { id: "pool_b", type: "maxpool2d", params: { kernel_size: 2, stride: 2 }, position: { x: 80 + INCEPTION_SPACING * 4, y: 80 + INCEPTION_BRANCH_Y * 3 } },
    { id: "conv_1x1_d", type: "conv2d", params: { in_channels: 64, out_channels: 32, kernel_size: 1, stride: 1, padding: 0 }, position: { x: 80 + INCEPTION_SPACING * 5, y: 80 + INCEPTION_BRANCH_Y * 3 } },
    { id: "relu_d", type: "relu", params: {}, position: { x: 80 + INCEPTION_SPACING * 6, y: 80 + INCEPTION_BRANCH_Y * 3 } },
    // Concat (chain: ab, cd, final). All branches output 7×7; channels: 64+128+32+32=256
    { id: "concat_ab", type: "concat", params: { dim: 1 }, position: { x: 80 + INCEPTION_SPACING * 10, y: 160 } },
    { id: "concat_cd", type: "concat", params: { dim: 1 }, position: { x: 80 + INCEPTION_SPACING * 10, y: 360 } },
    { id: "concat_out", type: "concat", params: { dim: 1 }, position: { x: 80 + INCEPTION_SPACING * 11, y: 280 } },
    // Classifier (7×7×256 = 12544)
    { id: "flatten_1", type: "flatten", params: {}, position: { x: 80 + INCEPTION_SPACING * 12, y: 280 } },
    { id: "fc", type: "linear", params: { in_features: 12544, out_features: 1000 }, position: { x: 80 + INCEPTION_SPACING * 13, y: 280 } },
    { id: "output_1", type: "output", params: {}, position: { x: 80 + INCEPTION_SPACING * 14, y: 280 } },
  ],
  edges: [
    { id: "e0", source: "input_1", sourceHandle: "out", target: "conv_stem", targetHandle: "in" },
    { id: "e1", source: "conv_stem", sourceHandle: "out", target: "relu_stem", targetHandle: "in" },
    { id: "e2", source: "relu_stem", sourceHandle: "out", target: "pool_stem", targetHandle: "in" },
    { id: "e3a", source: "pool_stem", sourceHandle: "out", target: "conv_1x1_a", targetHandle: "in" },
    { id: "e3b", source: "pool_stem", sourceHandle: "out", target: "conv_1x1_b", targetHandle: "in" },
    { id: "e3c", source: "pool_stem", sourceHandle: "out", target: "conv_1x1_c", targetHandle: "in" },
    { id: "e3d", source: "pool_stem", sourceHandle: "out", target: "pool_b", targetHandle: "in" },
    { id: "e4a", source: "conv_1x1_a", sourceHandle: "out", target: "relu_a", targetHandle: "in" },
    { id: "e4b", source: "conv_1x1_b", sourceHandle: "out", target: "relu_b1", targetHandle: "in" },
    { id: "e4c", source: "conv_1x1_c", sourceHandle: "out", target: "relu_c1", targetHandle: "in" },
    { id: "e4d", source: "pool_b", sourceHandle: "out", target: "conv_1x1_d", targetHandle: "in" },
    { id: "e5b", source: "relu_b1", sourceHandle: "out", target: "conv_3x3_b", targetHandle: "in" },
    { id: "e5b1", source: "conv_3x3_b", sourceHandle: "out", target: "relu_b2", targetHandle: "in" },
    { id: "e5c", source: "relu_c1", sourceHandle: "out", target: "conv_5x5_c", targetHandle: "in" },
    { id: "e5c1", source: "conv_5x5_c", sourceHandle: "out", target: "relu_c2", targetHandle: "in" },
    { id: "e5d", source: "conv_1x1_d", sourceHandle: "out", target: "relu_d", targetHandle: "in" },
    { id: "e5a", source: "relu_a", sourceHandle: "out", target: "pool_a", targetHandle: "in" },
    { id: "e5b2", source: "relu_b2", sourceHandle: "out", target: "pool_b2", targetHandle: "in" },
    { id: "e5c2", source: "relu_c2", sourceHandle: "out", target: "pool_c", targetHandle: "in" },
    { id: "e6a", source: "pool_a", sourceHandle: "out", target: "concat_ab", targetHandle: "in_a" },
    { id: "e6b", source: "pool_b2", sourceHandle: "out", target: "concat_ab", targetHandle: "in_b" },
    { id: "e6c", source: "pool_c", sourceHandle: "out", target: "concat_cd", targetHandle: "in_a" },
    { id: "e6d", source: "relu_d", sourceHandle: "out", target: "concat_cd", targetHandle: "in_b" },
    { id: "e7", source: "concat_ab", sourceHandle: "out", target: "concat_out", targetHandle: "in_a" },
    { id: "e8", source: "concat_cd", sourceHandle: "out", target: "concat_out", targetHandle: "in_b" },
    { id: "e9", source: "concat_out", sourceHandle: "out", target: "flatten_1", targetHandle: "in" },
    { id: "e10", source: "flatten_1", sourceHandle: "out", target: "fc", targetHandle: "in" },
    { id: "e11", source: "fc", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "GoogLeNet / Inception (Szegedy et al.)",
    created_at: new Date().toISOString(),
    description: "Inception module: stem (Conv→ReLU→MaxPool) feeds four parallel branches—1×1 only, 1×1→3×3, 1×1→5×5, MaxPool→1×1—then Concat along channels. Key idea: multi-scale feature extraction in parallel.",
  },
};

// ---------------------------------------------------------------------------
// Paper: Exploring the Limits of Transfer Learning (Raffel et al. — T5)
// ---------------------------------------------------------------------------

const T5_SPACING = 280;
const T5_ENC_MAIN_Y = 380; // encoder main path (ln1, attn, ln2, linear)
const T5_ENC_ADD_Y = 280; // enc_add1, enc_add2 elevated for visible skips
const T5_DEC_MAIN_Y = 160; // decoder main path
const T5_DEC_ADD_Y = 80; // dec_add1, dec_add2 elevated for visible skips
/** T5-style encoder-decoder: elevated Add blocks so skip connections are visible. */
const T5_PAPER_GRAPH: GraphSchema = {
  version: "1.0",
  nodes: [
    // Shared input pipeline
    { id: "text_input_1", type: "text_input", params: { batch_size: 1, seq_len: 128 }, position: { x: 80, y: 450 } },
    { id: "text_embed_1", type: "text_embedding", params: { vocab_size: 32128, embedding_dim: 128 }, position: { x: 80 + T5_SPACING, y: 450 } },
    { id: "pos_embed_1", type: "positional_embedding", params: { d_model: 128, max_len: 512 }, position: { x: 80 + T5_SPACING * 2, y: 450 } },
    // Encoder block — Add blocks elevated
    { id: "enc_ln1", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 80 + T5_SPACING * 2, y: T5_ENC_MAIN_Y } },
    { id: "enc_attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 80 + T5_SPACING * 3, y: T5_ENC_MAIN_Y } },
    { id: "enc_add1", type: "add", params: {}, position: { x: 80 + T5_SPACING * 4, y: T5_ENC_ADD_Y } },
    { id: "enc_ln2", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 80 + T5_SPACING * 2, y: T5_ENC_MAIN_Y } },
    { id: "enc_linear1", type: "linear", params: { in_features: 128, out_features: 512 }, position: { x: 80 + T5_SPACING * 3, y: T5_ENC_MAIN_Y } },
    { id: "enc_gelu", type: "activation", params: { activation: "gelu" }, position: { x: 80 + T5_SPACING * 4, y: T5_ENC_MAIN_Y } },
    { id: "enc_linear2", type: "linear", params: { in_features: 512, out_features: 128 }, position: { x: 80 + T5_SPACING * 5, y: T5_ENC_MAIN_Y } },
    { id: "enc_add2", type: "add", params: {}, position: { x: 80 + T5_SPACING * 6, y: T5_ENC_ADD_Y } },
    // Decoder block — Add blocks elevated
    { id: "dec_ln1", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 80 + T5_SPACING * 6, y: T5_DEC_MAIN_Y } },
    { id: "dec_attn", type: "attention", params: { embed_dim: 128, num_heads: 4 }, position: { x: 80 + T5_SPACING * 7, y: T5_DEC_MAIN_Y } },
    { id: "dec_add1", type: "add", params: {}, position: { x: 80 + T5_SPACING * 8, y: T5_DEC_ADD_Y } },
    { id: "dec_ln2", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 80 + T5_SPACING * 7, y: T5_DEC_MAIN_Y } },
    { id: "dec_linear1", type: "linear", params: { in_features: 128, out_features: 512 }, position: { x: 80 + T5_SPACING * 8, y: T5_DEC_MAIN_Y } },
    { id: "dec_gelu", type: "activation", params: { activation: "gelu" }, position: { x: 80 + T5_SPACING * 9, y: T5_DEC_MAIN_Y } },
    { id: "dec_linear2", type: "linear", params: { in_features: 512, out_features: 128 }, position: { x: 80 + T5_SPACING * 10, y: T5_DEC_MAIN_Y } },
    { id: "dec_add2", type: "add", params: {}, position: { x: 80 + T5_SPACING * 11, y: T5_DEC_ADD_Y } },
    { id: "dec_ln3", type: "layernorm", params: { normalized_shape: 128 }, position: { x: 80 + T5_SPACING * 10, y: 20 } },
    { id: "output_1", type: "output", params: {}, position: { x: 80 + T5_SPACING * 11, y: 20 } },
  ],
  edges: [
    { id: "e0a", source: "text_input_1", sourceHandle: "out", target: "text_embed_1", targetHandle: "in" },
    { id: "e0b", source: "text_embed_1", sourceHandle: "out", target: "pos_embed_1", targetHandle: "in" },
    { id: "e1", source: "pos_embed_1", sourceHandle: "out", target: "enc_ln1", targetHandle: "in" },
    { id: "e2", source: "pos_embed_1", sourceHandle: "out", target: "enc_add1", targetHandle: "in_a" },
    { id: "e3", source: "enc_ln1", sourceHandle: "out", target: "enc_attn", targetHandle: "in" },
    { id: "e4", source: "enc_attn", sourceHandle: "out", target: "enc_add1", targetHandle: "in_b" },
    { id: "e5", source: "enc_add1", sourceHandle: "out", target: "enc_ln2", targetHandle: "in" },
    { id: "e6", source: "enc_ln2", sourceHandle: "out", target: "enc_linear1", targetHandle: "in" },
    { id: "e7", source: "enc_ln2", sourceHandle: "out", target: "enc_add2", targetHandle: "in_a" },
    { id: "e8", source: "enc_linear1", sourceHandle: "out", target: "enc_gelu", targetHandle: "in" },
    { id: "e9", source: "enc_gelu", sourceHandle: "out", target: "enc_linear2", targetHandle: "in" },
    { id: "e10", source: "enc_linear2", sourceHandle: "out", target: "enc_add2", targetHandle: "in_b" },
    { id: "e11", source: "enc_add2", sourceHandle: "out", target: "dec_ln1", targetHandle: "in" },
    { id: "e12", source: "enc_add2", sourceHandle: "out", target: "dec_add1", targetHandle: "in_a" },
    { id: "e13", source: "dec_ln1", sourceHandle: "out", target: "dec_attn", targetHandle: "in" },
    { id: "e14", source: "dec_attn", sourceHandle: "out", target: "dec_add1", targetHandle: "in_b" },
    { id: "e15", source: "dec_add1", sourceHandle: "out", target: "dec_ln2", targetHandle: "in" },
    { id: "e16", source: "dec_ln2", sourceHandle: "out", target: "dec_linear1", targetHandle: "in" },
    { id: "e17", source: "dec_ln2", sourceHandle: "out", target: "dec_add2", targetHandle: "in_a" },
    { id: "e18", source: "dec_linear1", sourceHandle: "out", target: "dec_gelu", targetHandle: "in" },
    { id: "e19", source: "dec_gelu", sourceHandle: "out", target: "dec_linear2", targetHandle: "in" },
    { id: "e20", source: "dec_linear2", sourceHandle: "out", target: "dec_add2", targetHandle: "in_b" },
    { id: "e21", source: "dec_add2", sourceHandle: "out", target: "dec_ln3", targetHandle: "in" },
    { id: "e22", source: "dec_ln3", sourceHandle: "out", target: "output_1", targetHandle: "in" },
  ],
  metadata: {
    name: "T5 (Raffel et al.)",
    created_at: new Date().toISOString(),
    description: "T5 encoder-decoder: Text Input → Embedding → Position Embedding → Encoder block (Self-Attention + FFN with GELU) → Decoder block (Masked Self-Attention + FFN). T5 uses relative position and cross-attention in full decoder; here we show the block structure.",
  },
};

const LEVELS = [
  { level_number: 1, section: "challenges" as const, name: "Connect input to output", description: LEVEL_1_GRAPH.metadata.description!, task: "Create a feed forward network using the flatten and linear layer", graph_json: LEVEL_1_GRAPH, solution_graph_json: LEVEL_1_SOLUTION },
  { level_number: 2, section: "challenges" as const, name: "Add activation", description: LEVEL_2_GRAPH.metadata.description!, task: "Add a ReLU activation between the Linear layer and the Output (Input → Flatten → Linear → Activation → Output)", graph_json: LEVEL_2_GRAPH, solution_graph_json: LEVEL_2_SOLUTION },
  { level_number: 3, section: "challenges" as const, name: "Simple CNN", description: LEVEL_3_GRAPH.metadata.description!, task: "Build a small CNN: Input → Conv2D (e.g. 32 filters) → Activation → Flatten → Linear → Output", graph_json: LEVEL_3_GRAPH, solution_graph_json: LEVEL_3_SOLUTION },
  { level_number: 4, section: "challenges" as const, name: "Dropout regularization", description: LEVEL_4_GRAPH.metadata.description!, task: "Add Dropout between Linear and Output (Input → Flatten → Linear → Dropout → Output)", graph_json: LEVEL_4_GRAPH, solution_graph_json: LEVEL_4_SOLUTION },
  { level_number: 5, section: "challenges" as const, name: "LayerNorm and Attention", description: LEVEL_5_GRAPH.metadata.description!, task: "Build a path with LayerNorm then Attention (Input → LayerNorm → Attention → Output). Use 3D input or Embedding first.", graph_json: LEVEL_5_GRAPH, solution_graph_json: LEVEL_5_SOLUTION },
  { level_number: 6, section: "challenges" as const, name: "Residual with Add", description: LEVEL_6_GRAPH.metadata.description!, task: "Build one residual step: (1) Input → Flatten → Linear(784, 784) → Add. (2) Flatten → Add (skip path). So Add gets two inputs: from Linear and from Flatten. Connect both input ports of the Add block, then Add → Output. Both wires into Add must be shape [B, 784].", graph_json: LEVEL_6_GRAPH, solution_graph_json: LEVEL_6_SOLUTION },
  { level_number: 7, section: "challenges" as const, name: "Two-layer MLP", description: LEVEL_7_GRAPH.metadata.description!, task: "Build a two-layer feedforward network: Input → Flatten → Linear(784, 256) → ReLU → Linear(256, 128) → Output", graph_json: LEVEL_7_GRAPH, solution_graph_json: LEVEL_7_SOLUTION },
  { level_number: 8, section: "challenges" as const, name: "CNN with MaxPool", description: LEVEL_8_GRAPH.metadata.description!, task: "Build a CNN with pooling: Input → Conv2D (e.g. 32 filters) → ReLU → MaxPool2D (2×2) → Flatten → Linear → Output", graph_json: LEVEL_8_GRAPH, solution_graph_json: LEVEL_8_SOLUTION },
  { level_number: 9, section: "challenges" as const, name: "BatchNorm", description: LEVEL_9_GRAPH.metadata.description!, task: "Add BatchNorm between Linear and ReLU: Input → Flatten → Linear → BatchNorm → ReLU → Output", graph_json: LEVEL_9_GRAPH, solution_graph_json: LEVEL_9_SOLUTION },
  {
    level_number: 10,
    section: "papers" as const,
    paper_category: "language",
    name: "Attention is All You Need (Vaswani et al.)",
    description: TRANSFORMER_PAPER_GRAPH.metadata.description!,
    task: [
      "Vaswani et al., NeurIPS 2017. The Transformer relies entirely on self-attention without recurrence or convolution.",
      "• Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T / √d_k)V",
      "• Multi-Head Attention runs h attention heads in parallel (d_model = h × d_k); this model uses 4 heads, d_model=128.",
      "• Each encoder layer has two sublayers: (1) Multi-Head Self-Attention + residual & LayerNorm, (2) Position-wise FFN (two linear layers with ReLU) + residual & LayerNorm.",
      "• Input embeddings are combined with positional encodings (here: learned Positional Embedding); then stacked encoder layers produce the representation.",
    ].join("\n"),
    graph_json: TRANSFORMER_PAPER_GRAPH,
    solution_graph_json: TRANSFORMER_PAPER_GRAPH,
  },
  {
    level_number: 11,
    section: "papers" as const,
    paper_category: "vision",
    name: "ImageNet Classification with Deep CNNs (Krizhevsky et al.)",
    description: ALEXNET_PAPER_GRAPH.metadata.description!,
    task: [
      "Krizhevsky, Sutskever & Hinton, NeurIPS 2012. AlexNet revolutionized image classification with deep CNNs on ImageNet.",
      "• Architecture: 5 conv layers (96, 256, 384, 384, 256 filters) + 3 fully-connected (4096, 4096, 1000). ReLU after every conv and FC.",
      "• Conv1: 11×11, stride 4. Conv2–5: 5×5 and 3×3. The paper used MaxPool (3×3, stride 2) after conv1, conv2, and conv5.",
      "• Dropout (p=0.5) on the first two FC layers to reduce overfitting. FC8 outputs 1000-way ImageNet logits.",
      "• Key contributions: ReLU (faster than tanh), overlapping pooling, dropout, data augmentation, and training on two GPUs.",
    ].join("\n"),
    graph_json: ALEXNET_PAPER_GRAPH,
    solution_graph_json: ALEXNET_PAPER_GRAPH,
  },
  {
    level_number: 12,
    section: "papers" as const,
    paper_category: "vision",
    name: "Deep Residual Learning for Image Recognition (He et al.)",
    description: RESNET_PAPER_GRAPH.metadata.description!,
    task: [
      "He et al., CVPR 2016. ResNet uses residual (skip) connections to train very deep networks.",
      "• Residual block: F(x) = Conv→BN→ReLU→Conv→BN; output = x + F(x). The Add block sums identity and learned residual.",
      "• Skip connections let gradients flow directly through, avoiding vanishing gradients in 100+ layer networks.",
      "• Architecture: stem (Conv 7×7 or 3×3, BN, ReLU) → stacked residual blocks → global avg pool → FC. Here: 2 blocks, Flatten instead of avg pool.",
      "• BatchNorm after every conv (before ReLU). ReLU after each residual Add.",
    ].join("\n"),
    graph_json: RESNET_PAPER_GRAPH,
    solution_graph_json: RESNET_PAPER_GRAPH,
  },
  {
    level_number: 13,
    section: "papers" as const,
    paper_category: "language",
    name: "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.)",
    description: BERT_PAPER_GRAPH.metadata.description!,
    task: [
      "Devlin et al., NAACL 2019. BERT is an encoder-only Transformer pre-trained with Masked LM and NSP.",
      "• Architecture: identical to Transformer encoder. Token + Position embeddings → stacked encoder blocks.",
      "• Each block: Multi-Head Self-Attention (bidirectional) + Add & Norm → FFN (Linear → GELU → Linear) + Add & Norm.",
      "• BERT uses GELU (not ReLU) in the FFN. GELU(x) = x·Φ(x) where Φ is the Gaussian CDF; smoother than ReLU.",
      "• BERT-base: 12 layers, d_model=768, 12 heads. Output [CLS] or token representations for downstream tasks.",
    ].join("\n"),
    graph_json: BERT_PAPER_GRAPH,
    solution_graph_json: BERT_PAPER_GRAPH,
  },
  {
    level_number: 14,
    section: "papers" as const,
    paper_category: "language",
    name: "Improving Language Understanding by Generative Pre-Training (GPT)",
    description: GPT_PAPER_GRAPH.metadata.description!,
    task: [
      "Radford et al., 2018 (GPT-1); GPT-2 (2019), GPT-3 (2020). GPT is a decoder-only Transformer for autoregressive language modeling.",
      "• Causal masking: each token attends only to itself and previous tokens (no lookahead). Enables next-token prediction.",
      "• Architecture: same block as BERT (LayerNorm → Attention → Add → LayerNorm → FFN (GELU) → Add) but attention is masked. No encoder; no [CLS]/[SEP].",
      "• Pre-norm: LayerNorm is applied before each sublayer (attention and FFN), then residual Add. Stabilizes deep decoder stacks.",
      "• GPT-2: 12–48 layers, d_model=768–1600, vocab 50257. Output is projected to vocab for next-token logits; training is maximum likelihood over sequences.",
    ].join("\n"),
    graph_json: GPT_PAPER_GRAPH,
    solution_graph_json: GPT_PAPER_GRAPH,
  },
  {
    level_number: 15,
    section: "papers" as const,
    paper_category: "vision",
    name: "Gradient-Based Learning Applied to Document Recognition (LeNet-5)",
    description: LENET5_PAPER_GRAPH.metadata.description!,
    task: [
      "LeCun et al., 1998. LeNet-5 is the classic CNN for digit recognition (e.g. MNIST).",
      "• Architecture: Conv(6, 5×5) → tanh → MaxPool(2×2) → Conv(16, 5×5) → tanh → MaxPool(2×2) → Flatten → FC(120) → tanh → FC(84) → tanh → FC(10).",
      "• The original paper used tanh (and sigmoid at the output); we use tanh to match the paper. Subsampling used 2×2 average pooling; here MaxPool(2×2).",
      "• Key idea: alternating conv and pooling reduces spatial size; then fully connected layers produce class logits.",
    ].join("\n"),
    graph_json: LENET5_PAPER_GRAPH,
    solution_graph_json: LENET5_PAPER_GRAPH,
  },
  {
    level_number: 16,
    section: "papers" as const,
    paper_category: "vision",
    name: "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)",
    description: VGG_PAPER_GRAPH.metadata.description!,
    task: [
      "Simonyan & Zisserman, ICLR 2015. VGG uses only 3×3 convolutions stacked in blocks, with 2×2 MaxPool between blocks.",
      "• Each block: multiple Conv(3×3) + ReLU, then MaxPool(2×2). Deeper blocks use more channels (64 → 128 → 256 → 512 in full VGG-16).",
      "• Here: two blocks (64→64 and 128→128 filters), then Flatten and three FC layers (4096→4096→1000) with dropout.",
      "• Key contribution: depth with small receptive fields improves accuracy; 3×3 convs are the building block of modern CNNs.",
    ].join("\n"),
    graph_json: VGG_PAPER_GRAPH,
    solution_graph_json: VGG_PAPER_GRAPH,
  },
  {
    level_number: 17,
    section: "papers" as const,
    paper_category: "vision",
    name: "Going Deeper with Convolutions (Szegedy et al.)",
    description: INCEPTION_PAPER_GRAPH.metadata.description!,
    task: [
      "Szegedy et al., CVPR 2015 (GoogLeNet). Inception uses parallel branches with different receptive fields, then concatenates along the channel dimension.",
      "• Inception module: (1) 1×1 only, (2) 1×1 → 3×3, (3) 1×1 → 5×5, (4) 3×3 MaxPool → 1×1. All branches run in parallel and Concat merges their outputs.",
      "• 1×1 convs reduce channels before expensive 3×3 and 5×5 convs (bottleneck). Multi-scale features improve representation.",
      "• Full GoogLeNet stacks many Inception modules; auxiliary classifiers help training. Here: stem + one Inception module + classifier.",
    ].join("\n"),
    graph_json: INCEPTION_PAPER_GRAPH,
    solution_graph_json: INCEPTION_PAPER_GRAPH,
  },
  {
    level_number: 18,
    section: "papers" as const,
    paper_category: "language",
    name: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
    description: T5_PAPER_GRAPH.metadata.description!,
    task: [
      "Raffel et al., JMLR 2020. T5 frames all NLP tasks as text-to-text: input text → output text. Encoder-decoder Transformer with shared vocabulary.",
      "• Encoder: bidirectional self-attention (like BERT). Processes the input fully before the decoder runs.",
      "• Decoder: causal (masked) self-attention for autoregressive generation, plus cross-attention to encoder hidden states. Uses GELU in the FFN.",
      "• T5 uses relative position embeddings (we show learned positional embedding). Same block structure for encoder and decoder; full model stacks many layers.",
    ].join("\n"),
    graph_json: T5_PAPER_GRAPH,
    solution_graph_json: T5_PAPER_GRAPH,
  },
];

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
    .upsert(LEVELS, { onConflict: "level_number" })
    .select("id, level_number, name");

  if (error) {
    console.error("Seed failed:", error.message);
    process.exit(1);
  }

  console.log("Levels seeded successfully:", data);
}

main();
