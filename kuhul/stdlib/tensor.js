/**
 * @fileoverview Tensor utilities for KUHUL.
 *
 * Provides a high-level Tensor class wrapping a flat Float32Array with
 * shape metadata and common factory / manipulation methods.
 *
 * @module kuhul/stdlib/tensor
 */

// ------------------------------------------------------------------ //
// Tensor
// ------------------------------------------------------------------ //

/**
 * A multi-dimensional tensor backed by a Float32Array.
 */
export class Tensor {
  /**
   * @param {string}      elementType - e.g. "float32"
   * @param {number[]}    shape       - Dimension sizes
   * @param {Float32Array} buffer     - Flat data buffer
   */
  constructor(elementType, shape, buffer) {
    this.elementType = elementType;
    this.shape       = shape;
    this.buffer      = buffer;
  }

  /**
   * Create a Tensor with uninitialized (zero) data.
   *
   * @param {string}   elementType
   * @param {number[]} shape
   * @returns {Tensor}
   */
  static create(elementType, shape) {
    const size = shape.reduce((a, d) => a * d, 1);
    return new Tensor(elementType, shape, new Float32Array(size));
  }

  /**
   * Create a Tensor filled with zeros.
   *
   * @param {number[]} shape
   * @returns {Tensor}
   */
  static zeros(shape) {
    return Tensor.create('float32', shape);
  }

  /**
   * Create a Tensor filled with ones.
   *
   * @param {number[]} shape
   * @returns {Tensor}
   */
  static ones(shape) {
    const t = Tensor.create('float32', shape);
    t.buffer.fill(1);
    return t;
  }

  /**
   * Create a Tensor filled with a specific scalar value.
   *
   * @param {number[]} shape
   * @param {number}   value
   * @returns {Tensor}
   */
  static fill(shape, value) {
    const t = Tensor.create('float32', shape);
    t.buffer.fill(value);
    return t;
  }

  /**
   * Create a Tensor from an existing flat array or Float32Array.
   *
   * @param {number[]|Float32Array} data
   * @param {number[]}              shape
   * @returns {Tensor}
   */
  static from(data, shape) {
    const buffer = data instanceof Float32Array ? data : new Float32Array(data);
    return new Tensor('float32', shape, buffer);
  }

  // ---------------------------------------------------------------- //
  // Instance methods
  // ---------------------------------------------------------------- //

  /**
   * Total number of elements.
   * @returns {number}
   */
  get numel() { return this.buffer.length; }

  /**
   * Number of dimensions.
   * @returns {number}
   */
  get ndim() { return this.shape.length; }

  /**
   * Reshape to new shape (must have same total number of elements).
   *
   * @param {number[]} newShape
   * @returns {Tensor}
   */
  reshape(newShape) {
    const newSize = newShape.reduce((a, d) => a * d, 1);
    if (newSize !== this.numel) {
      throw new Error(`Cannot reshape tensor of size ${this.numel} to shape [${newShape}]`);
    }
    return new Tensor(this.elementType, newShape, this.buffer.slice());
  }

  /**
   * Get the value at a flat index.
   *
   * @param {number} index
   * @returns {number}
   */
  at(index) { return this.buffer[index]; }

  /**
   * Set the value at a flat index.
   *
   * @param {number} index
   * @param {number} value
   */
  set(index, value) { this.buffer[index] = value; }

  toString() {
    return `Tensor<${this.elementType}>[${this.shape.join('×')}]`;
  }
}
