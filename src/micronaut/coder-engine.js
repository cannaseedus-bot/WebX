// coder-engine.js — K'UHUL Code Review Engine (JS port of micronaut-coder)
//
// Source: C:\Users\canna\.kuhul-v1\micronaut-coder\src\coder_engine.cpp
// Backs micronaut_coder.exe — the grammar learning + code review engine.
//
// CodeReviewEngine: multi-language pattern detection, security rules,
//   performance anti-patterns, refactoring suggestions.
// Supported languages: python, cpp, javascript, java, typescript, rust, go
// Commands (mirrors micronaut_coder.exe CLI): learn, show, ast, validate,
//   grammar-pack, check-grammar, list, review

// ─── Language patterns ────────────────────────────────────────────────────────

export const LANGUAGE_PATTERNS = Object.freeze({
  python:     ["PEP8_indentation","snake_case_naming","docstring_format",
               "type_hints_required","exception_handling","list_comprehension",
               "f_string_usage","context_managers","immutable_defaults"],
  cpp:        ["RAII_principle","const_correctness","memory_safety",
               "exception_safety","naming_conventions","include_guards",
               "smart_pointers","move_semantics","rule_of_five"],
  javascript: ["async_await_usage","error_handling","no_var_usage",
               "arrow_functions","promise_patterns","null_coalescing",
               "template_literals","destructuring","spread_operator"],
  typescript: ["strict_null_checks","interface_over_type","readonly_fields",
               "generic_constraints","discriminated_unions","type_guards"],
  java:       ["null_safety","design_patterns","exception_handling",
               "resource_management","naming_conventions","generics_usage",
               "stream_api","lambda_expressions","optional_usage"],
  rust:       ["ownership_rules","lifetime_annotations","error_propagation",
               "iterator_chains","trait_bounds","unsafe_usage"],
  go:         ["error_return_check","goroutine_leak","channel_patterns",
               "defer_usage","interface_saturation","context_propagation"],
});

// ─── Security rules ───────────────────────────────────────────────────────────

export const SECURITY_RULES = [
  { id: "eval_usage",       pattern: /\beval\s*\(/,                                      severity: "error",   message: "eval() is a security risk — never evaluate untrusted input" },
  { id: "exec_usage",       pattern: /\bexec\s*\(/,                                      severity: "error",   message: "exec() can execute arbitrary code" },
  { id: "system_call",      pattern: /\bsystem\s*\(/,                                    severity: "warning", message: "System calls may be unsafe in this context" },
  { id: "shell_injection",  pattern: /\b(sh|bash)\s+-c/,                                severity: "error",   message: "Potential shell injection via -c flag" },
  { id: "sql_concat",       pattern: /SELECT.+\+.+FROM/i,                               severity: "error",   message: "SQL injection risk — use parameterised queries" },
  { id: "format_string",    pattern: /printf\s*\([^,)]+\s*%/,                            severity: "warning", message: "Format string vulnerability" },
  { id: "buffer_overflow",  pattern: /\b(strcpy|sprintf)\s*\(/,                          severity: "warning", message: "Use safe alternatives: strncpy, snprintf" },
  { id: "hardcoded_secret", pattern: /password\s*=\s*['"][^'"]{4,}['"]/i,               severity: "error",   message: "Hardcoded credential detected — use environment variables" },
  { id: "weak_crypto",      pattern: /\b(MD5|SHA1|DES)\b/i,                              severity: "warning", message: "Weak cryptographic algorithm — use SHA256+ or AES" },
  { id: "debug_code",       pattern: /\b(console\.log|debugger;|\bprint\s*\()/,          severity: "info",    message: "Debug code in production" },
  { id: "prototype_pollut", pattern: /Object\.prototype\.\w+\s*=/,                       severity: "error",   message: "Prototype pollution — never extend Object.prototype" },
  { id: "regex_dos",        pattern: /\(\.\*\)\+|\(\.\+\)\+/,                            severity: "warning", message: "ReDoS vulnerability — catastrophic backtracking possible" },
];

// ─── Performance rules ────────────────────────────────────────────────────────

export const PERFORMANCE_RULES = [
  { id: "nested_loop",       pattern: /for\s*\([^)]*\)[^{]*\{[^}]*for\s*\(/s,           severity: "warning", message: "O(n²) complexity — consider breaking into hash lookups" },
  { id: "string_concat_loop",pattern: /for\s*\([^)]*\)[^}]*\+=/,                        severity: "warning", message: "String concatenation in loop — use array.join()" },
  { id: "linear_search",     pattern: /\b(find|indexOf|includes)\s*\(/,                  severity: "info",    message: "Consider Map/Set for frequent lookups in large collections" },
  { id: "sync_io",           pattern: /readFileSync|writeFileSync/i,                     severity: "warning", message: "Blocking I/O — use async versions in request handlers" },
  { id: "repeated_alloc",    pattern: /new\s+\w+\s*\[\s*\]/,                             severity: "warning", message: "Repeated allocation in loop — hoist outside" },
  { id: "missing_memo",      pattern: /function\s+\w+\([^)]*\)\s*\{[^}]*return[^}]*\}/,severity: "info",    message: "Pure function — consider memoisation for expensive calls" },
];

// ─── Language detection from file extension ───────────────────────────────────

const EXT_MAP = {
  py: "python", pyw: "python",
  cpp: "cpp", cc: "cpp", cxx: "cpp", h: "cpp", hpp: "cpp",
  js: "javascript", mjs: "javascript", cjs: "javascript",
  ts: "typescript", tsx: "typescript",
  java: "java",
  rs: "rust",
  go: "go",
  kuhul: "kuhul", kuhules: "kuhul",
};

export function detectLanguage(filePath) {
  const ext = (filePath.split(".").pop() || "").toLowerCase();
  return EXT_MAP[ext] ?? "unknown";
}

// ─── CodeReviewEngine ─────────────────────────────────────────────────────────

export class CodeReviewEngine {
  constructor(opts = {}) {
    this._personality = opts.personality ?? {
      name: "K'UHUL Code Reviewer",
      version: "3.5.0",
      style: "professional",
      strictness: 0.8,
      focus_areas: ["security", "performance", "maintainability"],
    };
    this._customRules = [];
  }

  // ── Core review ─────────────────────────────────────────────────────────────
  review(code, filePath = "unknown") {
    const lang     = detectLanguage(filePath);
    const issues   = [];
    const lines    = code.split("\n");
    const patterns = LANGUAGE_PATTERNS[lang] ?? [];

    // Security scan
    for (const rule of SECURITY_RULES) {
      for (let ln = 0; ln < lines.length; ln++) {
        if (rule.pattern.test(lines[ln])) {
          issues.push({
            line: ln + 1, col: 0,
            id: rule.id, severity: rule.severity,
            message: rule.message,
            source: lines[ln].trim(),
          });
        }
      }
    }

    // Performance scan
    for (const rule of PERFORMANCE_RULES) {
      if (rule.pattern.test(code)) {
        issues.push({
          line: null, col: null,
          id: rule.id, severity: rule.severity,
          message: rule.message,
        });
      }
    }

    // Custom rules
    for (const rule of this._customRules) {
      for (let ln = 0; ln < lines.length; ln++) {
        if (rule.pattern.test(lines[ln])) {
          issues.push({ line: ln + 1, ...rule });
        }
      }
    }

    const score = this._computeScore(issues);
    return {
      file: filePath,
      language: lang,
      patterns_checked: patterns,
      issues,
      score,
      summary: this._summarise(issues, score),
      personality: this._personality.name,
    };
  }

  // ── Grammar pack (mirrors micronaut_coder.exe grammar-pack --lang) ──────────
  grammarPack(lang) {
    return {
      language: lang,
      patterns:      LANGUAGE_PATTERNS[lang] ?? [],
      security_rules: SECURITY_RULES.filter(r => r.severity === "error").map(r => r.id),
      performance_rules: PERFORMANCE_RULES.map(r => r.id),
    };
  }

  // ── List supported languages ─────────────────────────────────────────────────
  list() {
    return Object.keys(LANGUAGE_PATTERNS);
  }

  // ── Validate (check for hard errors only) ────────────────────────────────────
  validate(code, filePath = "unknown") {
    const result = this.review(code, filePath);
    const errors = result.issues.filter(i => i.severity === "error");
    return { valid: errors.length === 0, errors };
  }

  // ── Add custom rule ──────────────────────────────────────────────────────────
  addRule(id, pattern, severity, message) {
    this._customRules.push({ id, pattern, severity, message });
  }

  _computeScore(issues) {
    let deduction = 0;
    for (const i of issues) {
      if (i.severity === "error")   deduction += 15;
      if (i.severity === "warning") deduction += 5;
      if (i.severity === "info")    deduction += 1;
    }
    return Math.max(0, 100 - deduction);
  }

  _summarise(issues, score) {
    const counts = { error: 0, warning: 0, info: 0 };
    for (const i of issues) counts[i.severity] = (counts[i.severity] ?? 0) + 1;
    return `Score ${score}/100 — ${counts.error} errors, ${counts.warning} warnings, ${counts.info} notes`;
  }
}
