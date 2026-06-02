#pragma once

#include "vm_instruction.h"
#include <vector>

class WaveVM
{
public:
    WaveVM() 
        : field_(nullptr), patterns_(MAX_PATTERNS), pattern_count_(0) {}
    
    void set_field(std::vector<SHNodeCPU>* field) { field_ = field; }
    
    // Execution
    bool execute(std::vector<Instruction>& program);
    void step_execution();
    
    // Pattern library
    void store_pattern(const Pattern& p);
    void extract_current_pattern(Pattern& p);
    uint32_t classify_node(uint32_t node_id);
    
    // Input/Output
    void inject_signal(const InputSignal& sig);
    float read_coherence();
    float read_band_energy(uint32_t node_id, uint32_t band);
    
    // Accessors
    const std::vector<Pattern>& get_patterns() const { return patterns_; }
    uint32_t get_pattern_count() const { return pattern_count_; }
    std::vector<SHNodeCPU>* get_field() { return field_; }
    
private:
    std::vector<SHNodeCPU>* field_;
    std::vector<Pattern> patterns_;
    uint32_t pattern_count_;
    
    VMContext context_;
    
    // Internal opcodes
    void op_inject(const Instruction& ins);
    void op_propagate(const Instruction& ins);
    void op_collapse(const Instruction& ins);
    void op_compare(const Instruction& ins);
    void op_route(const Instruction& ins);
    void op_memory(const Instruction& ins);
};
