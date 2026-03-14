// K'UHUL++ Standard Library — GPU Memory Transfer
// Utilities for uploading Float32Array data to a GPUBuffer and downloading
// results back to the CPU.  Requires a WebGPU-capable environment.

// ------------------------------------------------------------------ //
// Types
// ------------------------------------------------------------------ //

/** Minimal WebGPU interface subset used by this module */
interface MinimalGPUDevice {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    queue: {
        writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: ArrayBuffer): void;
    };
    createCommandEncoder(): GPUCommandEncoder;
}

/** GPUBuffer with optional label */
interface GPUBuffer {
    label?:    string;
    size:      number;
    mapAsync(mode: number): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

interface GPUBufferDescriptor {
    size:             number;
    usage:            number;
    mappedAtCreation?: boolean;
    label?:            string;
}

interface GPUCommandEncoder {
    copyBufferToBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void;
    finish(): unknown;
}

// GPU usage flags (subset of the WebGPU spec)
const GPUBufferUsage = {
    STORAGE:   0x0080,
    COPY_SRC:  0x0004,
    COPY_DST:  0x0008,
    MAP_READ:  0x0001,
    MAP_WRITE: 0x0002,
};

// ------------------------------------------------------------------ //
// Upload
// ------------------------------------------------------------------ //

/**
 * Upload a Float32Array to a GPU storage buffer.
 *
 * @param device - WebGPU device
 * @param data   - CPU-side float data
 * @param label  - Optional debug label
 * @returns GPUBuffer containing the uploaded data
 *
 * @example
 * const adapter = await navigator.gpu.requestAdapter();
 * const device  = await adapter.requestDevice();
 * const buf     = await uploadToGPU(device, new Float32Array([1, 2, 3]));
 */
export async function uploadToGPU(
    device: MinimalGPUDevice,
    data:   Float32Array,
    label?: string,
): Promise<GPUBuffer> {
    const buffer = device.createBuffer({
        size:             data.byteLength,
        usage:            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
        label,
    });

    const mapped = buffer.getMappedRange();
    new Float32Array(mapped).set(data);
    buffer.unmap();

    return buffer;
}

// ------------------------------------------------------------------ //
// Download
// ------------------------------------------------------------------ //

/**
 * Download the contents of a GPU storage buffer into a Float32Array.
 *
 * @param device - WebGPU device
 * @param buffer - Source GPUBuffer (must have COPY_SRC usage)
 * @returns CPU-side Float32Array copy of the buffer contents
 *
 * @example
 * const result = await downloadFromGPU(device, outputBuffer);
 * console.log(result[0]);
 */
export async function downloadFromGPU(
    device: MinimalGPUDevice,
    buffer: GPUBuffer,
): Promise<Float32Array> {
    // Create a staging buffer with MAP_READ usage
    const staging = device.createBuffer({
        size:  buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    (device as any).queue.submit([(encoder as any).finish()]);

    // Map the staging buffer for reading
    await staging.mapAsync(GPUBufferUsage.MAP_READ);
    const mapped = staging.getMappedRange();
    const result = new Float32Array(mapped.byteLength / 4);
    result.set(new Float32Array(mapped));
    staging.unmap();
    staging.destroy();

    return result;
}

// ------------------------------------------------------------------ //
// Convenience: allocate an empty output buffer
// ------------------------------------------------------------------ //

/**
 * Create an empty GPU buffer for compute shader output.
 *
 * @param device    - WebGPU device
 * @param numFloats - Number of float32 elements
 * @param label     - Optional debug label
 */
export function createOutputBuffer(
    device:    MinimalGPUDevice,
    numFloats: number,
    label?:    string,
): GPUBuffer {
    return device.createBuffer({
        size:  numFloats * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label,
    });
}
