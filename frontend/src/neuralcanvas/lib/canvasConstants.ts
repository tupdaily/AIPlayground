// ---------------------------------------------------------------------------
// NeuralCanvas — shared UI scale for blocks and shape displays
// ---------------------------------------------------------------------------
// v3: Blocks are now auto-sized white cards. Scale set to 1.0 for full
// readability. No more fixed-size constraint.

/** Uniform scale for block size (blocks, handles). Full size. */
export const CANVAS_UI_SCALE = 1.0;

/** Default block width in px (auto-height). Blocks can override via definition.width. */
export const BLOCK_BASE_WIDTH = 260;

/** Legacy compat — kept for any callers but no longer constraining height. */
export const BLOCK_FIXED_SIZE = 260;

/** Scale for shape labels on wires. */
export const SHAPE_LABEL_SCALE = 1.0;
