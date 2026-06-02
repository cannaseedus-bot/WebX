#include "svg_compiler.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

SVG3DCompiler::SVG3DCompiler()
    : ast_({ 9, 162, 0.99f, 0.05f, {}, {}, {} })
{
}

SVG3DCompiler::~SVG3DCompiler()
{
}

bool SVG3DCompiler::parse(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        error_msg_ = "Failed to open file: " + filename;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string xml_text = buffer.str();

    return parse_xml(xml_text);
}

bool SVG3DCompiler::parse_xml(const std::string& xml_text)
{
    // Extract field config
    if (!extract_field_config(xml_text))
        return false;

    // Extract nodes
    if (!extract_nodes(xml_text))
        return false;

    // Extract edges
    if (!extract_edges(xml_text))
        return false;

    // Build ID → index map
    if (!build_id_map())
        return false;

    return true;
}

bool SVG3DCompiler::extract_field_config(const std::string& xml_text)
{
    // Find <domain> tag
    size_t domain_start = xml_text.find("<domain");
    if (domain_start == std::string::npos)
    {
        // Use defaults
        return true;
    }

    size_t domain_end = xml_text.find("/>", domain_start);
    if (domain_end == std::string::npos)
        return false;

    std::string domain_tag = xml_text.substr(domain_start, domain_end - domain_start + 2);

    ast_.decay = extract_float(domain_tag, "decay", 0.99f);
    ast_.coupling = extract_float(domain_tag, "coupling", 0.05f);

    // Find <meta> for band count (sh_l → bands = (l+1)^2)
    size_t meta_start = xml_text.find("<meta");
    if (meta_start != std::string::npos)
    {
        size_t meta_end = xml_text.find("/>", meta_start);
        std::string meta_tag = xml_text.substr(meta_start, meta_end - meta_start + 2);
        
        int sh_l = extract_int(meta_tag, "sh_l", 2);
        ast_.bands = (sh_l + 1) * (sh_l + 1);
    }

    return true;
}

bool SVG3DCompiler::extract_nodes(const std::string& xml_text)
{
    size_t graph_start = xml_text.find("<graph");
    size_t graph_end = xml_text.find("</graph>", graph_start);

    if (graph_start == std::string::npos || graph_end == std::string::npos)
        return true;  // Empty graph is OK

    std::string graph_section = xml_text.substr(graph_start, graph_end - graph_start);

    // Find all <node .../> tags
    size_t pos = 0;
    while ((pos = graph_section.find("<node", pos)) != std::string::npos)
    {
        size_t tag_end = graph_section.find("/>", pos);
        if (tag_end == std::string::npos)
            break;

        std::string node_tag = graph_section.substr(pos, tag_end - pos + 2);

        SVG3DNode node;
        node.id = extract_attr(node_tag, "id");
        node.op = extract_attr(node_tag, "type");

        if (node.id.empty())
        {
            error_msg_ = "Node missing id attribute";
            return false;
        }

        // Store all attributes
        node.attrs["pos"] = extract_attr(node_tag, "pos");
        node.attrs["band"] = extract_attr(node_tag, "band");
        node.attrs["amplitude"] = extract_attr(node_tag, "amplitude");
        node.attrs["phase"] = extract_attr(node_tag, "phase");

        ast_.node_list.push_back(node);
        pos = tag_end + 2;
    }

    return true;
}

bool SVG3DCompiler::extract_edges(const std::string& xml_text)
{
    size_t graph_start = xml_text.find("<graph");
    size_t graph_end = xml_text.find("</graph>", graph_start);

    if (graph_start == std::string::npos || graph_end == std::string::npos)
        return true;

    std::string graph_section = xml_text.substr(graph_start, graph_end - graph_start);

    // Find all <edge .../> tags
    size_t pos = 0;
    while ((pos = graph_section.find("<edge", pos)) != std::string::npos)
    {
        size_t tag_end = graph_section.find("/>", pos);
        if (tag_end == std::string::npos)
            break;

        std::string edge_tag = graph_section.substr(pos, tag_end - pos + 2);

        SVG3DEdge edge;
        edge.from = extract_attr(edge_tag, "from");
        edge.to = extract_attr(edge_tag, "to");
        edge.weight = extract_float(edge_tag, "weight", 1.0f);

        if (edge.from.empty() || edge.to.empty())
        {
            error_msg_ = "Edge missing from/to attributes";
            return false;
        }

        ast_.edges.push_back(edge);
        pos = tag_end + 2;
    }

    return true;
}

std::string SVG3DCompiler::extract_attr(const std::string& tag, const std::string& attr_name)
{
    std::string search = attr_name + "=\"";
    size_t start = tag.find(search);
    if (start == std::string::npos)
        return "";

    start += search.length();
    size_t end = tag.find("\"", start);
    if (end == std::string::npos)
        return "";

    return tag.substr(start, end - start);
}

int SVG3DCompiler::extract_int(const std::string& tag, const std::string& attr_name, int default_val)
{
    std::string str_val = extract_attr(tag, attr_name);
    if (str_val.empty())
        return default_val;

    return std::stoi(str_val);
}

float SVG3DCompiler::extract_float(const std::string& tag, const std::string& attr_name, float default_val)
{
    std::string str_val = extract_attr(tag, attr_name);
    if (str_val.empty())
        return default_val;

    return std::stof(str_val);
}

bool SVG3DCompiler::build_id_map()
{
    for (int i = 0; i < (int)ast_.node_list.size(); i++)
    {
        ast_.id_to_index[ast_.node_list[i].id] = i;
    }
    return true;
}

bool SVG3DCompiler::topological_sort(std::vector<int>& out_order)
{
    // Build adjacency list from edges
    std::vector<std::vector<int>> adj(ast_.node_list.size());
    std::vector<int> in_degree(ast_.node_list.size(), 0);

    for (const auto& edge : ast_.edges)
    {
        auto it_from = ast_.id_to_index.find(edge.from);
        auto it_to = ast_.id_to_index.find(edge.to);

        if (it_from == ast_.id_to_index.end() || it_to == ast_.id_to_index.end())
        {
            error_msg_ = "Edge references non-existent node";
            return false;
        }

        int from_idx = it_from->second;
        int to_idx = it_to->second;

        adj[from_idx].push_back(to_idx);
        in_degree[to_idx]++;
    }

    // Kahn's algorithm
    std::vector<int> queue;
    for (int i = 0; i < (int)in_degree.size(); i++)
    {
        if (in_degree[i] == 0)
            queue.push_back(i);
    }

    while (!queue.empty())
    {
        int u = queue.back();
        queue.pop_back();
        out_order.push_back(u);

        for (int v : adj[u])
        {
            in_degree[v]--;
            if (in_degree[v] == 0)
                queue.push_back(v);
        }
    }

    if (out_order.size() != ast_.node_list.size())
    {
        error_msg_ = "Graph contains cycle";
        return false;
    }

    return true;
}

bool SVG3DCompiler::compile(std::vector<Instruction>& out_program)
{
    // Get topological order
    std::vector<int> topo_order;
    if (!topological_sort(topo_order))
        return false;

    // Compile each node in order
    for (int node_idx : topo_order)
    {
        const SVG3DNode& node = ast_.node_list[node_idx];

        Instruction instr = { };

        if (node.op == "inject")
        {
            instr.op = OP_INJECT;
            instr.a = node_idx;  // node ID
            
            // Extract band, amplitude, phase from attrs
            if (node.attrs.count("band"))
                instr.b = std::stoi(node.attrs.at("band"));
            if (node.attrs.count("amplitude"))
                instr.f0 = std::stof(node.attrs.at("amplitude"));
            if (node.attrs.count("phase"))
                instr.f1 = std::stof(node.attrs.at("phase"));
        }
        else if (node.op == "probe")
        {
            instr.op = OP_COLLAPSE;
            instr.a = node_idx;
            if (node.attrs.count("band"))
                instr.b = std::stoi(node.attrs.at("band"));
        }
        else if (node.op == "memory")
        {
            instr.op = OP_MEMORY;
            instr.a = 1;  // store flag
        }
        else if (node.op == "compare")
        {
            instr.op = OP_COMPARE;
            instr.a = 0;  // pattern index
            instr.f0 = 0.5f;  // threshold
        }
        else if (node.op == "propagate")
        {
            instr.op = OP_PROPAGATE;
            if (node.attrs.count("steps"))
                instr.a = std::stoi(node.attrs.at("steps"));
            else
                instr.a = 1;
        }
        else if (node.op == "route")
        {
            instr.op = OP_ROUTE;
        }
        else
        {
            error_msg_ = "Unknown opcode: " + node.op;
            return false;
        }

        out_program.push_back(instr);
    }

    // Add halt
    Instruction halt = { };
    halt.op = OP_HALT;
    out_program.push_back(halt);

    return true;
}
