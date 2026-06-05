// geodesic-fold-sphere.js — Deterministic semantic atlas on a geodesic sphere
//
// Bridges domain-specific fold placement with the PGGTF geometry stack:
//   GeodesicTensor  — sparse k-NN graph of geodesic distances
//   FoldTensor      — GEO lane (3) caches each fold's nearest neighbors
//   SphericalManifold — all geometric ops (expMap, logMap, parallelTransport)
//
// Domain descriptor drives placement: same descriptor → same sphere position.
// Required for K'UHUL collapse invariant (same input → same output).
//
// Domain descriptor shape:
//   { name, glyph, binding, fold, entropy }
//   name    — fold identifier
//   glyph   — 'Sek'|'Pop'|'Wo'|"Ch'en"|'Yax'|'Xul'
//   binding — 'π' (immutable→north hemisphere) | 'τ' (temporal→south hemisphere)
//   fold    — sw.khl category: ai|ui|runtime|os|tapes|dns|mesh|
//             security|trainer|rlhf|quantum|atomic|gram
//   entropy — float; 0.21 = canonical collapsed shell (K'UHUL invariant)
//
// Training integration:
//   projectEmbedding(vec3) → sphere point  (token embedding → position)
//   attentionWeights(queryFold, candidateFolds) → geodesic-distance scores
//   toArcWeights() → sparse { (src,dst): quality * exp(-entropy) } for ARC bias

import { GeodesicTensor }   from './geodesic-tensor.js';
import { FoldTensor, SCXQ2_LANES } from './fold-tensor.js';
import { SphericalManifold } from '../xcfe/geo-weights.js';

// ── Sector tables ─────────────────────────────────────────────────────────────

// sw.khl 13 fold categories → θ sectors (2π / 13 each)
const FOLD_SECTORS = {
    ai:0, ui:1, runtime:2, os:3, tapes:4, dns:5, mesh:6,
    security:7, trainer:8, rlhf:9, quantum:10, atomic:11, gram:12,
};
const N_SECTORS = 13;

// Six canonical glyphs → φ offset within hemisphere (normalized 0→1)
const GLYPH_PHI = {
    'Sek': 0.10, 'Pop': 0.28, 'Wo': 0.46,
    "Ch'en": 0.64, 'Yax': 0.82, 'Xul': 1.00,
};

// ── GeodesicFoldSphere ────────────────────────────────────────────────────────

export class GeodesicFoldSphere {
    /**
     * @param {object}  opts
     * @param {number}  opts.curvature   — sphere curvature (default 0.1 → radius ≈ 3.16)
     * @param {number}  opts.k_nearest   — neighbor count for GeodesicTensor (default 8)
     * @param {number}  opts.n_folds     — FoldTensor fold count (default 512)
     */
    constructor(opts = {}) {
        this.manifold  = new SphericalManifold(opts.curvature ?? 0.1);
        this.k_nearest = opts.k_nearest ?? 8;

        // Fold registry: name → { domain, position[3], neighbors[] }
        this.folds = new Map();

        // Built lazily when folds are committed
        this._G = null;
        this._F = new FoldTensor(opts.n_folds ?? 512, 128, 256);
        this._dirty = false;
    }

    // ── Deterministic placement ───────────────────────────────────────────────

    /**
     * Map a domain descriptor to a deterministic point on the sphere.
     * Hemisphere split:  π-binding → north (φ ∈ [0, π/4])
     *                    τ-binding → south (φ ∈ [3π/4, π])
     */
    hashToPosition(domain = {}) {
        const { name = '', glyph = 'Sek', binding = 'τ', fold = 'runtime', entropy = 0.21 } = domain;

        // θ — fold category sector + name offset within sector
        const sectorIdx  = FOLD_SECTORS[fold] ?? (this._hash(fold) % N_SECTORS);
        const sectorW    = (2 * Math.PI) / N_SECTORS;
        const theta      = sectorIdx * sectorW + (this._hash(name) / 0xFFFFFFFF) * sectorW;

        // φ — binding hemisphere; glyph modulates within hemisphere
        const glyphBase  = GLYPH_PHI[glyph] ?? (this._hash(glyph) / 0xFFFFFFFF);
        const phi        = binding === 'π'
            ? glyphBase * (Math.PI / 4)
            : Math.PI - glyphBase * (Math.PI / 4);

        // Cartesian on sphere surface (radius from manifold)
        const r = this.manifold.radius;
        const pos = [
            r * Math.sin(phi) * Math.cos(theta),
            r * Math.sin(phi) * Math.sin(theta),
            r * Math.cos(phi),
        ];

        return { theta, phi, entropy, cartesian: this.manifold.project(pos) };
    }

    /**
     * Register a fold by domain descriptor.
     * Returns the fold record with its deterministic sphere position.
     */
    placeFold(domain = {}) {
        const name = domain.name ?? 'unnamed';
        const coords = this.hashToPosition(domain);
        const record = { name, ...domain, ...coords };
        this.folds.set(name, record);
        this._dirty = true;
        return record;
    }

    // ── Graph + tensor cache ──────────────────────────────────────────────────

    /**
     * Build or rebuild the GeodesicTensor (k-NN graph) from current folds.
     * Writes neighbor distances into FoldTensor GEO lane (3).
     * Call after all folds are placed, or when _dirty.
     */
    commit() {
        const entries = Array.from(this.folds.values());
        const n = entries.length;
        if (n === 0) return this;

        const rows = [], cols = [], weights = [];

        for (let i = 0; i < n; i++) {
            // Compute distances from fold i to all others
            const dists = [];
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                dists.push({
                    j,
                    d: this.manifold.geodesicDistance(entries[i].cartesian, entries[j].cartesian),
                });
            }
            dists.sort((a, b) => a.d - b.d);
            const neighbors = dists.slice(0, this.k_nearest);

            for (const { j, d } of neighbors) {
                rows.push(i); cols.push(j); weights.push(d);
            }

            // Write to FoldTensor GEO lane — channel layout:
            //   [0 .. k-1]   → geodesic distances to k nearest neighbors
            //   [k .. 2k-1]  → neighbor fold indices (as float)
            const lane = SCXQ2_LANES.GEO;
            for (let k = 0; k < neighbors.length; k++) {
                this._F.set(i, lane, k,            neighbors[k].d);
                this._F.set(i, lane, k + this.k_nearest, neighbors[k].j);
            }
            // Channel 2*k: entropy of this fold
            this._F.set(i, lane, 2 * this.k_nearest, entries[i].entropy ?? 0.21);
        }

        this._G = new GeodesicTensor(n, rows, cols, weights);
        this._dirty = false;
        return this;
    }

    // ── Training integration ──────────────────────────────────────────────────

    /**
     * Project a token embedding (flat vector) onto the sphere surface.
     * Maps external embeddings (e.g. GPT-2 hidden states) to sphere positions.
     * Use this in geodesic_attention_bridge.py via the JS runtime bridge,
     * or mirror the logic directly in Python.
     */
    projectEmbedding(vec) {
        return this.manifold.project(vec);
    }

    /**
     * Geodesic attention scores between a query fold and candidate folds.
     * score = exp(-dist / radius)  →  nearby folds score high.
     * entropy scales the temperature: high entropy → flatter distribution.
     */
    attentionWeights(queryName, candidateNames) {
        if (this._dirty) this.commit();
        const q = this.folds.get(queryName);
        if (!q) return [];
        const r = this.manifold.radius;
        return candidateNames.map(name => {
            const c = this.folds.get(name);
            if (!c) return { name, score: 0 };
            const dist = this.manifold.geodesicDistance(q.cartesian, c.cartesian);
            const temp = q.entropy ?? 0.21;
            return { name, score: Math.exp(-dist / (r * temp)) };
        });
    }

    /**
     * Export sparse ARC weight matrix for geodesic_attention_bridge.py.
     * Format: Map<`${src},${dst}`, weight>
     * weight = quality * exp(-entropy)  — matches ARC accumulator pattern.
     */
    toArcWeights(quality = 1.0) {
        if (this._dirty) this.commit();
        const arc = new Map();
        const entries = Array.from(this.folds.values());
        const n = entries.length;
        for (let i = 0; i < n; i++) {
            const fi = entries[i];
            const lane = SCXQ2_LANES.GEO;
            for (let k = 0; k < this.k_nearest; k++) {
                const dist = this._F.get(i, lane, k);
                const j    = Math.round(this._F.get(i, lane, k + this.k_nearest));
                if (j < 0 || j >= n || dist === 0) continue;
                const entropy = fi.entropy ?? 0.21;
                arc.set(`${i},${j}`, quality * Math.exp(-entropy));
            }
        }
        return arc;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    get G() { if (this._dirty) this.commit(); return this._G; }
    get F() { return this._F; }

    visualize() {
        if (this._dirty) this.commit();
        return {
            folds: this.folds.size,
            radius: this.manifold.radius,
            curvature: this.manifold.curvature,
            k_nearest: this.k_nearest,
            geo_edges: this._G ? this._G.data.length : 0,
        };
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    /** djb2-style deterministic hash → unsigned 32-bit integer */
    _hash(str) {
        let h = 5381;
        for (let i = 0; i < str.length; i++) {
            h = ((h << 5) + h) ^ str.charCodeAt(i);
            h = h >>> 0;
        }
        return h;
    }
}

export function createGeodesicFoldSphere(opts) {
    return new GeodesicFoldSphere(opts);
}
