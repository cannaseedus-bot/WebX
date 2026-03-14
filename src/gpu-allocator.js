// GPU memory allocator implementation using SharedArrayBuffer

class GPUMemoryAllocator {
    constructor(size) {
        this.size = size;
        this.buffer = new SharedArrayBuffer(size);
        this.view = new Uint8Array(this.buffer);
        this.allocated = 0;
    }

    allocate(size) {
        if (this.allocated + size > this.size) {
            throw new Error('Insufficient memory');
        }
        const start = this.allocated;
        this.allocated += size;
        return this.view.subarray(start, start + size);
    }

    deallocate(size) {
        this.allocated -= size;
        if (this.allocated < 0) {
            this.allocated = 0; // Prevent negative allocation
        }
    }

    getBuffer() {
        return this.buffer;
    }
}

export default GPUMemoryAllocator;