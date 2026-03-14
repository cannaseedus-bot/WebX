/**
 * @fileoverview I/O operations for KUHUL programs.
 *
 * Provides file and stream I/O for the KUHUL runtime.  In browser
 * environments that lack Node.js `fs`, the implementation falls back to
 * in-memory storage.
 *
 * @module kuhul/stdlib/io
 */

// ------------------------------------------------------------------ //
// Optional Node.js imports (graceful fallback in browsers)
// ------------------------------------------------------------------ //

let fsPromises = null;
try {
  // Dynamic import avoids bundler errors in browser environments
  const mod = await import('fs/promises');
  fsPromises = mod;
} catch (_) {
  // Running in a browser or environment without Node.js fs
}

// ------------------------------------------------------------------ //
// KuhulIO
// ------------------------------------------------------------------ //

/** File and stream I/O utilities for KUHUL. */
export class KuhulIO {
  constructor() {
    /** In-memory fallback store. @type {Map<string, string>} */
    this._store = new Map();
  }

  /**
   * Read data from a file path (Node.js) or in-memory store (browser).
   *
   * @param {string} path
   * @returns {Promise<string>}
   */
  async read(path) {
    if (fsPromises) {
      return fsPromises.readFile(path, 'utf8');
    }
    if (this._store.has(path)) return this._store.get(path);
    throw new Error(`KuhulIO: file not found: ${path}`);
  }

  /**
   * Write data to a file path (Node.js) or in-memory store (browser).
   *
   * @param {string} path
   * @param {string} data
   * @returns {Promise<void>}
   */
  async write(path, data) {
    if (fsPromises) {
      return fsPromises.writeFile(path, data, 'utf8');
    }
    this._store.set(path, String(data));
  }

  /**
   * Delete a stored entry (in-memory store only).
   *
   * @param {string} path
   * @returns {boolean}
   */
  delete(path) { return this._store.delete(path); }

  /**
   * Check whether a path exists (in-memory store only).
   *
   * @param {string} path
   * @returns {boolean}
   */
  has(path) { return this._store.has(path); }
}
