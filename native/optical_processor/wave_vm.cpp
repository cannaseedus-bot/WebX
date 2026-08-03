#include "wave_vm.h"
#include <cmath>
#include <cstring>

bool WaveVM::execute(std::vector<Instruction>& program)
{
    if (!field_)
        return false;

    context_.field = field_;
    context_.patterns = &patterns_;
    context_.program = &program;
    context_.ip = 0;
    context_.last_result = 0.0f;
    context_.frame_count = 0;

    while (context_.ip < program.size())
    {
        const Instruction& ins = program[context_.ip];

        switch (ins.op)
        {
            case OP_INJECT:
                op_inject(ins);
                break;

            case OP_PROPAGATE:
                op_propagate(ins);
                break;

            case OP_COLLAPSE:
                op_collapse(ins);
                break;

            case OP_COMPARE:
                op_compare(ins);
                break;

            case OP_ROUTE:
                op_route(ins);
                continue;  // Skip increment for branching

            case OP_MEMORY:
                op_memory(ins);
                break;

            case OP_HALT:
                return true;

            default:
                return false;
        }

        context_.ip++;
    }

    return true;
}

void WaveVM::op_inject(const Instruction& ins)
{
    if (ins.a >= field_->size() || ins.b >= SH_BANDS)
        return;

    InputSignal sig;
    sig.node_id = ins.a;
    sig.band = ins.b;
    sig.amplitude = ins.f0;
    sig.phase = ins.f1;

    inject_signal(sig);
}

void WaveVM::op_propagate(const Instruction& ins)
{
    // Simple wave propagation step
    // (In real system: call optical_processor_.stepWavePropagation())
    
    uint32_t steps = (ins.a > 0) ? ins.a : 1;

    for (uint32_t s = 0; s < steps; s++)
    {
        for (size_t i = 0; i < field_->size(); i++)
        {
            auto& node = (*field_)[i];
            SHNodeCPU temp = node;

            for (int j = 0; j < SH_BANDS; j++)
            {
                float x = node.sh[j].x;
                float y = node.sh[j].y;

                // Cross-band coupling
                if (j + 1 < SH_BANDS)
                {
                    x += node.sh[(j + 1) % SH_BANDS].x * 0.05f;
                    y += node.sh[(j + 1) % SH_BANDS].y * 0.05f;
                }

                // Phase rotation
                float angle = (j + 1) * 0.02f;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);

                float nx = x * cos_a - y * sin_a;
                float ny = x * sin_a + y * cos_a;

                // Normalize
                float len = sqrtf(nx * nx + ny * ny);
                if (len > 1e-4f)
                    len = 1.0f / len;
                else
                    len = 1.0f;

                nx *= len;
                ny *= len;

                // Energy injection
                nx += 0.005f;
                ny += 0.003f;

                // Decay
                nx *= 0.99f;
                ny *= 0.99f;

                temp.sh[j].x = nx;
                temp.sh[j].y = ny;
            }

            node = temp;
        }

        context_.frame_count++;
    }
}

void WaveVM::op_collapse(const Instruction& ins)
{
    // Read coherence from specified node
    if (ins.a >= field_->size())
    {
        context_.last_result = 0.0f;
        return;
    }

    context_.last_result = compute_coherence(*field_);
}

void WaveVM::op_compare(const Instruction& ins)
{
    // Compare node vs pattern
    if (ins.a >= field_->size() || ins.b >= pattern_count_)
    {
        context_.last_result = 1e6f;  // Large distance
        return;
    }

    context_.last_result = pattern_distance(patterns_[ins.b], (*field_)[ins.a]);
}

void WaveVM::op_route(const Instruction& ins)
{
    // Branch if last_result < threshold
    if (context_.last_result < ins.f0)
    {
        context_.ip = ins.a;
    }
}

void WaveVM::op_memory(const Instruction& ins)
{
    // Store current pattern if a=1, otherwise load
    if (ins.a == 1 && pattern_count_ < MAX_PATTERNS)
    {
        extract_current_pattern(patterns_[pattern_count_]);
        pattern_count_++;
    }
}


void WaveVM::inject_signal(const InputSignal& sig)
{
    if (sig.node_id >= field_->size() || sig.band >= SH_BANDS)
        return;

    auto& node = (*field_)[sig.node_id];

    float c = cosf(sig.phase);
    float s = sinf(sig.phase);

    node.sh[sig.band].x += sig.amplitude * c;
    node.sh[sig.band].y += sig.amplitude * s;
}

void WaveVM::store_pattern(const Pattern& p)
{
    if (pattern_count_ < MAX_PATTERNS)
    {
        patterns_[pattern_count_] = p;
        pattern_count_++;
    }
}

void WaveVM::extract_current_pattern(Pattern& p)
{
    extract_signature(*field_, p);
    p.discovered_frame = context_.frame_count;
    p.stability = compute_coherence(*field_);
}

uint32_t WaveVM::classify_node(uint32_t node_id)
{
    if (node_id >= field_->size() || pattern_count_ == 0)
        return UINT32_MAX;

    float min_distance = 1e6f;
    uint32_t best_pattern = 0;

    for (uint32_t i = 0; i < pattern_count_; i++)
    {
        float d = pattern_distance(patterns_[i], (*field_)[node_id]);
        if (d < min_distance)
        {
            min_distance = d;
            best_pattern = i;
        }
    }

    return best_pattern;
}

float WaveVM::read_coherence()
{
    return compute_coherence(*field_);
}

float WaveVM::read_band_energy(uint32_t node_id, uint32_t band)
{
    if (node_id >= field_->size() || band >= SH_BANDS)
        return 0.0f;

    return ::read_band_energy((*field_)[node_id], band);
}
