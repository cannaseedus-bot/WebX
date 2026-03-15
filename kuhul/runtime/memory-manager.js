/**
 * @fileoverview Memory manager for the KUHUL runtime manifold M.
 *
 * Allocates, tracks, and frees typed Float32 memory blocks identified by
 * integer IDs.
 *
 * @module kuhul/runtime/memory-manager
 */

// ------------------------------------------------------------------ //
// MemoryBlock
// ------------------------------------------------------------------ //

/**
 * A single contiguous block of Float32 memory.
 * @typedef {{ id: number, buffer: Float32Array, size: number }} MemoryBlock
 */

// ------------------------------------------------------------------ //
// MemoryManager
// ------------------------------------------------------------------ //

/** Manages Float32 memory allocations on the KUHUL manifold. */
export class MemoryManager {
  constructor() {
    /** @type {Map<number, { id: number, buffer: Float32Array, size: number }>} */
    this._blocks  = new Map();
    this._nextId  = 1;
    this._totalAllocated = 0;
  }

  /**
   * Allocate a Float32 block of `size` elements.
   *
   * @param {number} size - Number of float32 elements
   * @returns {number} Block ID
   */
  allocate(size) {
    if (!Number.isInteger(size) || size <= 0) {
      throw new RangeError(`MemoryManager.allocate: size must be a positive integer, got ${size}`);
    }
    const id = this._nextId++;
    const buffer = new Float32Array(size);
    this._blocks.set(id, { id, buffer, size });
    this._totalAllocated += size;
    return id;
  }

  /**
   * Free a previously allocated block.
   *
   * @param {number} id - Block ID returned by `allocate`
   * @returns {boolean} True if the block existed and was freed
   */
  free(id) {
    const block = this._blocks.get(id);
    if (!block) return false;
    this._totalAllocated -= block.size;
    this._blocks.delete(id);
    return true;
  }

  /**
   * Retrieve a memory block by ID.
   *
   * @param {number} id
   * @returns {{ id: number, buffer: Float32Array, size: number }|null}
   */
  get(id) {
    return this._blocks.get(id) ?? null;
  }

  /**
   * Total number of float32 elements currently allocated.
   * @returns {number}
   */
  get totalAllocated() {
    return this._totalAllocated;
  }

  /**
   * Number of live memory blocks.
   * @returns {number}
   */
  get blockCount() {
    return this._blocks.size;
  }

  /** Free all blocks. */
  reset() {
    this._blocks.clear();
    this._totalAllocated = 0;
    this._nextId = 1;
  }
}
