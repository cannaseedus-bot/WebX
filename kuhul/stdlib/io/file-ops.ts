// K'UHUL++ Standard Library — File I/O
// Read and write .kuhul / .kpp source files using the Node.js fs/promises API.
// This module is Node.js-only; browser environments should use the GPU transfer API.

import { readFile, writeFile } from 'fs/promises';
import { resolve, extname }   from 'path';

// ------------------------------------------------------------------ //
// Constants
// ------------------------------------------------------------------ //

const KUHUL_EXTENSIONS = new Set(['.kuhul', '.kpp', '.kuh']);

// ------------------------------------------------------------------ //
// File operations
// ------------------------------------------------------------------ //

/**
 * Read a K'UHUL++ source file and return its contents as a UTF-8 string.
 *
 * @param filePath - Absolute or relative path to the .kuhul / .kpp file
 * @returns Source code string
 * @throws {Error} If the file cannot be read or has an unexpected extension
 *
 * @example
 * const source = await readKuhulFile('./programs/neural-layer.kuhul');
 */
export async function readKuhulFile(filePath: string): Promise<string> {
    const resolved = resolve(filePath);
    const ext = extname(resolved).toLowerCase();

    if (!KUHUL_EXTENSIONS.has(ext) && ext !== '') {
        console.warn(`readKuhulFile: unexpected extension "${ext}" for "${resolved}" — proceeding anyway`);
    }

    const content = await readFile(resolved, { encoding: 'utf8' });
    return content;
}

/**
 * Write K'UHUL++ source code to a file.
 *
 * @param filePath - Destination path (should end in .kuhul or .kpp)
 * @param content  - K'UHUL++ source code to write
 *
 * @example
 * await writeKuhulFile('./out/result.kuhul', sourceCode);
 */
export async function writeKuhulFile(filePath: string, content: string): Promise<void> {
    const resolved = resolve(filePath);
    await writeFile(resolved, content, { encoding: 'utf8' });
}

/**
 * Read a binary file (e.g. a compiled .wasm module) as a Uint8Array.
 *
 * @param filePath - Path to the binary file
 */
export async function readBinaryFile(filePath: string): Promise<Uint8Array> {
    const resolved = resolve(filePath);
    const buffer   = await readFile(resolved);
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

/**
 * Write a Uint8Array to a binary file.
 *
 * @param filePath - Destination path
 * @param data     - Binary data to write
 */
export async function writeBinaryFile(filePath: string, data: Uint8Array): Promise<void> {
    const resolved = resolve(filePath);
    await writeFile(resolved, Buffer.from(data));
}
