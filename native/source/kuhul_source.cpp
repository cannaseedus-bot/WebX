#include "kuhul_source.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace Kuhul::Source {

namespace {

void addUnique(std::vector<std::string>& values, const std::string& value) {
    if (!value.empty() &&
        std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string jsonString(const std::string& value);

std::string jsonArray(const std::vector<std::string>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << "\"" << jsonString(values[i]) << "\"";
    }
    out << "]";
    return out.str();
}

std::string jsonString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\') escaped += "\\\\";
        else if (character == '"') escaped += "\\\"";
        else if (character == '\n') escaped += "\\n";
        else escaped += character;
    }
    return escaped;
}

} // namespace

bool load(const std::string& path, Document& document) {
    document = Document{};
    document.path = path;

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        document.errors.push_back("cannot_open_file");
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    document.source = buffer.str();
    if (document.source.empty()) {
        document.errors.push_back("empty_source");
        return false;
    }

    std::istringstream lines(document.source);
    std::string line;
    bool inImports = false;
    bool inAlgebras = false;
    while (std::getline(lines, line)) {
        ++document.line_count;
        const std::string clean = trim(line);
        if (clean.find("imports") != std::string::npos) inImports = true;
        if (clean.find("algebras") != std::string::npos) inAlgebras = true;
        if (inImports) {
            const auto quote = clean.find('"');
            if (quote != std::string::npos) {
                const auto end = clean.find('"', quote + 1);
                if (end != std::string::npos) {
                    addUnique(document.imports, clean.substr(quote + 1, end - quote - 1));
                }
            }
            if (clean.find(']') != std::string::npos) inImports = false;
        }
        if (clean.find("fold:") != std::string::npos ||
            clean.find("folds:") != std::string::npos) {
            const auto marker = clean.find("folds:") != std::string::npos ? "folds:" : "fold:";
            const auto start = clean.find(marker) + std::string(marker).size();
            addUnique(document.folds, trim(clean.substr(start)));
        }
        if (inAlgebras || clean.find("algebras") != std::string::npos) {
            for (const char* domain : {"SymbolicExecution", "NeuralTraining",
                                       "Inference", "ShaderCompilation",
                                       "MicronautForging", "ProviderResolution",
                                       "SCXWorkingSet"}) {
                if (clean.find(domain) != std::string::npos) addUnique(document.domains, domain);
            }
            if ((clean == "}" || clean == "};") &&
                clean.find("algebras") == std::string::npos) {
                inAlgebras = false;
            }
        }
        if (clean.find("provider") != std::string::npos ||
            clean.find("capabilities") != std::string::npos) {
            addUnique(document.providers, clean);
        }
        for (const char* phase : {"Pop", "Wo", "Yax", "Sek", "Chen", "Xul"}) {
            if (clean.find(phase) != std::string::npos) addUnique(document.phases, phase);
        }
        for (const char character : clean) {
            if (character == '{') ++document.brace_depth;
            if (character == '}') {
                if (document.brace_depth == 0) {
                    document.errors.push_back("unmatched_closing_brace");
                } else {
                    --document.brace_depth;
                }
            }
        }
    }
    if (document.brace_depth != 0) document.errors.push_back("unclosed_brace");
    return document.valid();
}

std::string astJson(const Document& document) {
    std::ostringstream out;
    out << "{\n"
        << "  \"type\": \"KAST\",\n"
        << "  \"path\": \"" << jsonString(document.path) << "\",\n"
        << "  \"lines\": " << document.line_count << ",\n"
        << "  \"imports\": " << jsonArray(document.imports) << ",\n"
        << "  \"folds\": " << jsonArray(document.folds) << ",\n"
        << "  \"domains\": " << jsonArray(document.domains) << ",\n"
        << "  \"providers\": " << jsonArray(document.providers) << ",\n"
        << "  \"phases\": " << jsonArray(document.phases) << ",\n"
        << "  \"valid\": " << (document.valid() ? "true" : "false") << "\n"
        << "}";
    return out.str();
}

std::string analysis(const Document& document) {
    std::ostringstream out;
    out << "KAST analysis\n"
        << "  path:       " << document.path << "\n"
        << "  lines:      " << document.line_count << "\n"
        << "  imports:    " << document.imports.size() << "\n"
        << "  folds:      " << document.folds.size() << "\n"
        << "  domains:    " << document.domains.size() << "\n"
        << "  providers:  " << document.providers.size() << "\n"
        << "  phases:     " << document.phases.size() << "\n"
        << "  braces:     " << document.brace_depth << "\n"
        << "  valid:      " << (document.valid() ? "true" : "false") << "\n";
    for (const auto& error : document.errors) out << "  error:      " << error << "\n";
    return out.str();
}

std::string semanticPackage(const Document& document) {
    std::ostringstream out;
    out << "SCX-KUHUL-PACKAGE\n"
        << "version=1\n"
        << "source=" << document.path << "\n"
        << "valid=" << (document.valid() ? "true" : "false") << "\n"
        << "imports=" << document.imports.size() << "\n"
        << "folds=" << document.folds.size() << "\n"
        << "domains=" << document.domains.size() << "\n"
        << "providers=" << document.providers.size() << "\n"
        << "note=semantic package; provider bytecode is resolved during execution\n";
    return out.str();
}

std::string generatedCode(const Document& document, const std::string& target) {
    std::ostringstream out;
    out << "// K'UHUL generated host skeleton\n"
        << "// source: " << document.path << "\n"
        << "// target: " << target << "\n"
        << "// semantic authority remains the .kuhul contract\n\n";
    if (target == "cpp") {
        out << "#include \"runtime/phase_runtime.h\"\n"
            << "void execute_kuhul(RuntimeContext& context) {\n"
            << "    Pop(context); Wo(context); Yax(context); Sek(context); Chen(context); Xul(context);\n"
            << "}\n";
    } else if (target == "hlsl" || target == "wgsl" || target == "opencl") {
        out << "// Provider-specific kernel emission is deferred to the selected native provider.\n";
    } else {
        out << "// No emitter is registered for this target.\n";
    }
    return out.str();
}

bool writeFile(const std::string& path, const std::string& contents, std::string& error) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        error = "cannot_write_file";
        return false;
    }
    output << contents;
    if (!output) {
        error = "write_failed";
        return false;
    }
    return true;
}

} // namespace Kuhul::Source
