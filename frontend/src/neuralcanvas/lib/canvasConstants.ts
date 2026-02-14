// ---------------------------------------------------------------------------
// NeuralCanvas — shared UI scale for blocks and shape displays
// ---------------------------------------------------------------------------
// Use a single scale so blocks, wire labels, and shape badges stay proportional.
// All blocks (including Input/Output) use the same dimensions from BaseBlock.

/** Uniform scale for block size (blocks, handles). */
export const CANVAS_UI_SCALE = 0.72;

/** Fixed block width/height multiplier so every block is the same size. */
export const BLOCK_FIXED_SIZE = 88;

/** Scale for shape labels on wires (slightly smaller than blocks). */
export const SHAPE_LABEL_SCALE = 0.85;
