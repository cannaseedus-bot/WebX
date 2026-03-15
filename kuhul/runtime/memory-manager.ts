// K'UHUL++ Memory Manager
// Manages manifold M memory — typed tensor allocations that may reside in
// SharedArrayBuffer for zero-copy GPU access.

import type { KuhulType, DataType, TensorType, ScalarType } from '../ir/ir-types.js';

// Re-export for convenience
export type { KuhulType } from '../ir/ir-types.js';
export type { GeometricIR, IRInstruction } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// Allocation descriptor
// ------------------------------------------------------------------ //

export interface Allocation {
    id:     string;
    type:   KuhulType;
    buffer: SharedArrayBuffer | ArrayBuffer;
    view:   Float32Array | Float64Array | Int32Array | Uint32Array;
}

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

function totalElements(type: KuhulType): number {
    if (type.kind === 'tensor') {
        return (type as TensorType).shape.reduce((acc, d) => acc * d, 1) || 1;
    }
    return 1;
}

function bytesPerElement(dtype: DataType): number {
    switch (dtype) {
        case 'float32': return 4;
        case 'float64': return 8;
        case 'int32':
        case 'uint32':  return 4;
    }
}

function createView(
    buffer: SharedArrayBuffer | ArrayBuffer,
    dtype: DataType,
): Float32Array | Float64Array | Int32Array | Uint32Array {
    switch (dtype) {
        case 'float32': return new Float32Array(buffer);
        case 'float64': return new Float64Array(buffer);
        case 'int32':   return new Int32Array(buffer);
        case 'uint32':  return new Uint32Array(buffer);
    }
}

// ------------------------------------------------------------------ //
// MemoryManager
// ------------------------------------------------------------------ //

/**
 * Typed memory manager for K'UHUL++ tensor allocations.
 * Uses SharedArrayBuffer when available so that allocations can be
 * directly transferred to GPU workers without copying.
 *
 * @example
 * const mm = new MemoryManager();
 * const id = mm.allocate({ kind: 'tensor', dtype: 'float32', shape: [4, 4] });
 * mm.write(id, new Float32Array(16).fill(1));
 * const data = mm.read(id);
 */
export class MemoryManager {
    private allocations = new Map<string, Allocation>();
    private idCounter   = 0;
    private readonly useShared: boolean;

    /**
     * @param useShared - Prefer SharedArrayBuffer (default: auto-detect)
     */
    constructor(useShared?: boolean) {
        this.useShared = useShared ?? (typeof SharedArrayBuffer !== 'undefined');
    }

    // ---- Allocation ----

    /**
     * Allocate a typed buffer for a KUHUL type.
     *
     * @param type  - Tensor or scalar type descriptor
     * @param shape - Override shape (for dynamic sizes)
     * @returns Unique allocation id
     */
    allocate(type: KuhulType, shape?: number[]): string {
        const id = `m${this.idCounter++}`;

        let dtype: DataType = 'float32';
        let elements = 1;

        if (type.kind === 'tensor') {
            dtype    = type.dtype;
            elements = (shape ?? type.shape).reduce((a, b) => a * b, 1) || 1;
        } else if (type.kind === 'scalar') {
            dtype    = type.dtype;
            elements = 1;
        }

        const bytes  = elements * bytesPerElement(dtype);
        const buffer = this.useShared
            ? new SharedArrayBuffer(bytes)
            : new ArrayBuffer(bytes);

        const view = createView(buffer, dtype);
        this.allocations.set(id, { id, type, buffer, view });
        return id;
    }

    // ---- Read / Write ----

    /**
     * Read the typed view for an allocation.
     *
     * @param id - Allocation id returned by `allocate()`
     * @returns Typed array view or null if id is unknown
     */
    read(id: string): Float32Array | Float64Array | Int32Array | Uint32Array | null {
        return this.allocations.get(id)?.view ?? null;
    }

    /**
     * Write data into an allocation.
     *
     * @param id   - Allocation id
     * @param data - Source typed array (must match element count)
     */
    write(
        id: string,
        data: Float32Array | Float64Array | Int32Array | Uint32Array | number[],
    ): void {
        const alloc = this.allocations.get(id);
        if (!alloc) throw new Error(`MemoryManager: unknown allocation id "${id}"`);

        const view = alloc.view as any;
        const src  = Array.isArray(data) ? new Float32Array(data) : data;
        const len  = Math.min(view.length, src.length);
        for (let i = 0; i < len; i++) view[i] = (src as any)[i];
    }

    // ---- Lifecycle ----

    /**
     * Free an allocation, releasing its backing buffer.
     *
     * @param id - Allocation id to free
     */
    free(id: string): void {
        this.allocations.delete(id);
    }

    /** Free all allocations */
    freeAll(): void {
        this.allocations.clear();
    }

    // ---- Introspection ----

    /** Total number of live allocations */
    get size(): number { return this.allocations.size; }

    /** Retrieve the full allocation descriptor */
    getAllocation(id: string): Allocation | undefined {
        return this.allocations.get(id);
    }

    /** List all live allocation ids */
    listIds(): string[] {
        return [...this.allocations.keys()];
    }

    /** Total bytes currently allocated across all live allocations */
    get totalBytes(): number {
        let total = 0;
        for (const alloc of this.allocations.values()) {
            total += alloc.buffer.byteLength;
        }
        return total;
    }
}
