#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "vm_instruction.h"

// SVG-3D AST Node
struct SVG3DNode
{
    std::string id;
    std::string op;
    std::map<std::string, std::string> attrs;
};

// SVG-3D Edge
struct SVG3DEdge
{
    std::string from;
    std::string to;
    float weight;
};

// SVG-3D Program AST
struct SVG3DAST
{
    int bands;
    int nodes;
    float decay;
    float coupling;

    std::vector<SVG3DNode> node_list;
    std::vector<SVG3DEdge> edges;
    std::map<std::string, int> id_to_index;
};

// SVG → ISA Compiler
class SVG3DCompiler
{
public:
    SVG3DCompiler();
    ~SVG3DCompiler();

    // Parse SVG-3D XML file
    bool parse(const std::string& filename);

    // Compile to ISA
    bool compile(std::vector<Instruction>& out_program);

    // Accessors
    const SVG3DAST& get_ast() const { return ast_; }
    int bands() const { return ast_.bands; }
    int nodes() const { return ast_.nodes; }
    const std::string& error() const { return error_msg_; }

private:
    SVG3DAST ast_;
    std::string error_msg_;

    // XML parsing helpers
    bool parse_xml(const std::string& xml_text);
    bool extract_field_config(const std::string& xml_text);
    bool extract_nodes(const std::string& xml_text);
    bool extract_edges(const std::string& xml_text);

    // Attribute extraction
    std::string extract_attr(const std::string& line, const std::string& attr_name);
    int extract_int(const std::string& line, const std::string& attr_name, int default_val);
    float extract_float(const std::string& line, const std::string& attr_name, float default_val);

    // Compilation
    bool build_id_map();
    bool topological_sort(std::vector<int>& out_order);
};
