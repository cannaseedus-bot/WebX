// D12WebX CommandList - GPU command recording

/**
 * Records a sequence of GPU commands for deferred execution.
 * Commands are executed in the order they are recorded.
 */
class CommandList {
    constructor() {
        this.commands = [];
        this.closed = false;
    }

    /**
     * Record a compute dispatch command.
     * @param {number} x - Thread group count in X dimension
     * @param {number} y - Thread group count in Y dimension
     * @param {number} z - Thread group count in Z dimension
     */
    dispatch(x, y, z) {
        if (this.closed) throw new Error('CommandList is closed');
        this.commands.push({ type: 'dispatch', x, y, z, threads: x * y * z });
        return this;
    }

    /**
     * Record a KUHUL glyph execution command.
     * @param {string} glyph - KUHUL glyph symbol
     * @param {object} buffer - Target GPU buffer
     * @param {*} param - Glyph-specific parameter
     */
    execute(glyph, buffer, param) {
        if (this.closed) throw new Error('CommandList is closed');
        this.commands.push({ type: 'execute', glyph, buffer, param });
        return this;
    }

    /**
     * Record a memory copy command.
     * @param {object} src - Source buffer
     * @param {object} dst - Destination buffer
     * @param {number} size - Number of bytes to copy
     */
    copyBuffer(src, dst, size) {
        if (this.closed) throw new Error('CommandList is closed');
        this.commands.push({ type: 'copyBuffer', src, dst, size });
        return this;
    }

    /**
     * Record a buffer write command.
     * @param {object} buffer - Target GPU buffer
     * @param {ArrayBuffer|TypedArray} data - Data to write
     */
    writeBuffer(buffer, data) {
        if (this.closed) throw new Error('CommandList is closed');
        this.commands.push({ type: 'writeBuffer', buffer, data });
        return this;
    }

    /**
     * Close the command list for execution (no more commands can be added).
     */
    close() {
        this.closed = true;
        return this;
    }

    /**
     * Reset the command list for reuse.
     */
    reset() {
        this.commands = [];
        this.closed = false;
        return this;
    }

    /**
     * @returns {number} Number of recorded commands
     */
    get length() {
        return this.commands.length;
    }
}

export default CommandList;
