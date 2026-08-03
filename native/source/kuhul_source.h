#pragma once

#include <string>
#include <vector>

namespace Kuhul::Source {

struct Document {
    std::string path;
    std::string source;
    std::vector<std::string> imports;
    std::vector<std::string> folds;
    std::vector<std::string> domains;
    std::vector<std::string> providers;
    std::vector<std::string> phases;
    std::vector<std::string> errors;
    size_t line_count = 0;
    size_t brace_depth = 0;

    bool valid() const { return errors.empty() && !source.empty() && brace_depth == 0; }
};

bool load(const std::string& path, Document& document);
std::string astJson(const Document& document);
std::string analysis(const Document& document);
std::string semanticPackage(const Document& document);
std::string generatedCode(const Document& document, const std::string& target);
bool writeFile(const std::string& path, const std::string& contents, std::string& error);

} // namespace Kuhul::Source
