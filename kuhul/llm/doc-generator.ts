// K'UHUL++ Documentation Generator
// Generates Markdown documentation from a K'UHUL++ AST.
// Extracts comments, declarations, and glyph operations to produce
// structured API documentation.

import { NodeKind } from '../compiler/parser.js';
import type { ProgramNode, ASTNode } from '../compiler/parser.js';

// ------------------------------------------------------------------ //
// Documentation model
// ------------------------------------------------------------------ //

export interface DocEntry {
    name:          string;
    kind:          'tensor' | 'cluster' | 'model' | 'pipeline' | 'glyph';
    description:   string;
    type?:         string;
    glyph?:        string;
    children?:     DocEntry[];
}

export interface Documentation {
    title:       string;
    description: string;
    entries:     DocEntry[];
    /** Raw Markdown output */
    markdown:    string;
}

// ------------------------------------------------------------------ //
// DocGenerator
// ------------------------------------------------------------------ //

/**
 * Generates Markdown documentation from a K'UHUL++ AST.
 *
 * @example
 * const gen = new DocGenerator('My KUHUL Module');
 * const docs = gen.generateDocs(ast);
 * console.log(docs.markdown);
 */
export class DocGenerator {
    private readonly title: string;

    constructor(title = "K'UHUL++ Module Documentation") {
        this.title = title;
    }

    /**
     * Generate documentation from a parsed AST.
     *
     * @param ast  - AST from `parse()`
     * @returns Documentation object including Markdown
     */
    generateDocs(ast: ProgramNode): Documentation {
        const entries = this.extractEntries(ast.body);
        const markdown = this.renderMarkdown(entries);

        return {
            title:       this.title,
            description: `Auto-generated K'UHUL++ module documentation.`,
            entries,
            markdown,
        };
    }

    // ---- AST traversal ----

    private extractEntries(body: ASTNode[]): DocEntry[] {
        const entries: DocEntry[] = [];
        for (const node of body) {
            const entry = this.nodeToEntry(node);
            if (entry) entries.push(entry);
        }
        return entries;
    }

    private nodeToEntry(node: ASTNode): DocEntry | null {
        const n = node as any;
        switch (node.kind) {
            case NodeKind.TensorDecl:
                return {
                    name:        n.name,
                    kind:        'tensor',
                    description: this.describeInit(n.init),
                    type:        n.typeParams ? `Tensor<${n.typeParams.join(', ')}>` : 'Tensor',
                };

            case NodeKind.ClusterDecl:
                return {
                    name:        n.name,
                    kind:        'cluster',
                    description: `Tensor cluster containing ${n.body.length} declaration(s).`,
                    children:    this.extractEntries(n.body),
                };

            case NodeKind.ModelDecl:
                return {
                    name:        n.name,
                    kind:        'model',
                    description: `Neural model with ${n.body.length} member(s).`,
                    children:    this.extractEntries(n.body),
                };

            case NodeKind.PipelineDecl:
                return {
                    name:        n.name,
                    kind:        'pipeline',
                    description: `Compute pipeline with ${n.body.length} stage(s).`,
                    children:    this.extractEntries(n.body),
                };

            case NodeKind.GlyphOp:
                return {
                    name:        `${this.exprName(n.left)} ${n.glyph} ${this.exprName(n.right)}`,
                    kind:        'glyph',
                    glyph:       n.glyph,
                    description: this.glyphDescription(n.glyph),
                };

            default:
                return null;
        }
    }

    // ---- Helpers ----

    private describeInit(init: ASTNode): string {
        const n = init as any;
        switch (init.kind) {
            case NodeKind.NumberLiteral:  return `Numeric constant: ${n.value}`;
            case NodeKind.StringLiteral:  return `String value: "${n.value}"`;
            case NodeKind.PiExpr:         return `π expression: ${n.coefficient}π`;
            case NodeKind.ArrayLiteral:   return `Array of ${n.elements.length} element(s)`;
            case NodeKind.FunctionCall:   return `Result of ${n.callee}(...)`;
            case NodeKind.GlyphOp:        return `Glyph operation: ${n.glyph}`;
            default:                      return `Expression (${init.kind})`;
        }
    }

    private exprName(node: ASTNode): string {
        const n = node as any;
        switch (node.kind) {
            case NodeKind.Identifier:    return n.name;
            case NodeKind.NumberLiteral: return String(n.value);
            case NodeKind.StringLiteral: return `"${n.value}"`;
            default:                     return `(${node.kind})`;
        }
    }

    private glyphDescription(glyph: string): string {
        const map: Record<string, string> = {
            '⊗':  'Geometric product / tensor product',
            '⊕':  'Translation / bias addition in manifold M',
            '⊖':  'Difference / subtraction in manifold M',
            '⊛':  'Convolution in manifold M',
            '⊜':  'Identity element in M',
            '⊝':  'Complement / negation in M',
            '⊞':  'Union / element-wise addition',
            '⤍':  'Vector Encrypt — affine transform on vector field',
            '↻':  'Rotational Compression',
            '⟲':  'Spherical Loop transform',
            '∿':  'Torsion Field deformation',
            '⊙':  'Radial Projection',
            '≋':  'Wave Modulation',
        };
        return map[glyph] ?? `Unknown glyph operator "${glyph}"`;
    }

    // ---- Markdown rendering ----

    private renderMarkdown(entries: DocEntry[]): string {
        const lines: string[] = [
            `# ${this.title}`,
            '',
            `> Auto-generated K'UHUL++ module documentation.`,
            '',
        ];

        for (const entry of entries) {
            lines.push(...this.renderEntry(entry, 2));
        }

        return lines.join('\n');
    }

    private renderEntry(entry: DocEntry, depth: number): string[] {
        const h = '#'.repeat(depth);
        const badge = `\`${entry.kind}\``;
        const lines: string[] = [
            `${h} ${entry.name} ${badge}`,
            '',
            entry.description,
            '',
        ];

        if (entry.type) {
            lines.push(`**Type:** \`${entry.type}\``, '');
        }
        if (entry.glyph) {
            lines.push(`**Glyph:** ${entry.glyph}`, '');
        }
        if (entry.children && entry.children.length > 0) {
            lines.push('**Members:**', '');
            for (const child of entry.children) {
                lines.push(...this.renderEntry(child, depth + 1));
            }
        }

        return lines;
    }
}
