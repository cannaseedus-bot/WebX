// field_optimizer.hlsl — Win2D-parallel physics-based optimizer
//
// Replaces standard Adam with game-physics field dynamics:
//   attraction_well    → gravity toward loss minimum (⟁Grav⟁)
//   scroll_inertia     → momentum accumulation (Adam m1/m2)
//   wind_field         → learning rate directional push
//   navigation_force   → arrival steering (cosine decay near target)
//
// Each kernel dispatches per-parameter in parallel — Win2D/D3D11 compute.
// Thread group: 256 threads, each handles one float32 parameter.
//
// Fields map to K'UHUL gravity:
//   gravity_scale = 0.0 → antigravity (debug/telemetry, bypasses update)
//   gravity_scale = 1.0 → normal update (standard field dynamics)
//   gravity_scale = 2.0 → heavy (loss head, stronger attraction well)

cbuffer FieldParams : register(b0)
{
    float  lr;               // base learning rate
    float  beta1;            // scroll_inertia decay (default 0.9)
    float  beta2;            // scroll_inertia 2nd moment decay (default 0.999)
    float  epsilon;          // attraction_well epsilon (default 1e-8)
    float  weight_decay;     // wind_field strength (L2 regularization)
    float  grad_clip;        // max gradient magnitude (attraction_well bound)
    float  target_loss;      // navigation_force target (arrival behavior)
    float  current_loss;     // current loss for navigation_force
    float  gravity_scale;    // K'UHUL gravity field: 0=float, 1=normal, 2=heavy
    float  step;             // current training step (for bias correction)
    float  wind_direction;   // wind_field: 1.0=forward, -1.0=reverse (LR sign)
    float  inertia_friction; // scroll_inertia friction (arrival deceleration)
};

RWStructuredBuffer<float> params_buf  : register(u0);  // model weights
RWStructuredBuffer<float> grads_buf   : register(u1);  // gradients from backward
RWStructuredBuffer<float> m1_buf      : register(u2);  // scroll_inertia 1st moment
RWStructuredBuffer<float> m2_buf      : register(u3);  // scroll_inertia 2nd moment

// ── scroll_inertia_field ──────────────────────────────────────────────────────
// Simulates momentum with friction and decay (like scrolling physics).
// Replaces Adam m1/m2 accumulation.
float2 scroll_inertia(float g, float m1, float m2)
{
    // Exponential moving average (momentum = inertia with decay/friction)
    float new_m1 = beta1 * m1 + (1.0 - beta1) * g;
    float new_m2 = beta2 * m2 + (1.0 - beta2) * g * g;
    return float2(new_m1, new_m2);
}

// ── attraction_well_field ──────────────────────────────────────────────────────
// Gravity well that pulls parameters toward the loss minimum.
// Falloff: strong pull when far from minimum, gentle near convergence.
float attraction_well(float m1_hat, float m2_hat, float g_scale)
{
    // Well depth proportional to gravity_scale (⟁HeavyGrav⟁ = deeper well)
    float well_strength = g_scale;
    float update = well_strength * lr * m1_hat / (sqrt(m2_hat) + epsilon);

    // Clamp to grad_clip (attraction_well max force bound)
    return clamp(update, -grad_clip, grad_clip);
}

// ── wind_field ────────────────────────────────────────────────────────────────
// Constant directional force (learning rate as wind direction + strength).
// wind_direction: 1=forward training, -1=reverse (fine-tune reversal)
float wind_field(float param, float wd)
{
    // L2 regularization = wind pushing weights back toward origin
    return wind_direction * wd * weight_decay * param;
}

// ── navigation_force_field ────────────────────────────────────────────────────
// Arrival steering: full speed when far from target_loss, decelerate near it.
// Replaces cosine LR decay — smoother, physics-based arrival.
float navigation_force_scale()
{
    if (target_loss <= 0.0) return 1.0;  // no target set

    float dist = max(0.0, current_loss - target_loss);  // distance to goal
    float arrival_radius = target_loss * 2.0;            // start decelerating

    if (dist < arrival_radius && arrival_radius > 0.0) {
        // Arrival behavior: decelerate as we approach target
        // Friction increases: inertia_friction ∈ [0.5, 1.0]
        float t = dist / arrival_radius;  // 0 at target, 1 far away
        return max(0.1, t);  // minimum 10% speed — never fully stop
    }
    return 1.0;  // full speed when far from target
}

// ── Main optimizer kernel ─────────────────────────────────────────────────────

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)round(step * 0.0)) return;  // bounds check via dispatch size

    float g = grads_buf[i];
    float p = params_buf[i];

    // ── Antigravity bypass (⟁AntiGrav⟁: debug/telemetry nodes) ───────────────
    if (gravity_scale < 0.001)
    {
        // Float: no update, no gradient
        grads_buf[i] = 0.0;
        return;
    }

    // ── Gradient clamp (attraction_well bound — NaN guard) ────────────────────
    if (isnan(g) || isinf(g)) { grads_buf[i] = 0.0; return; }
    g = clamp(g, -grad_clip, grad_clip);

    // ── scroll_inertia: update momentum buffers ───────────────────────────────
    float2 inertia = scroll_inertia(g, m1_buf[i], m2_buf[i]);
    m1_buf[i] = inertia.x;
    m2_buf[i] = inertia.y;

    // Bias correction (Adam standard)
    float bc1 = 1.0 - pow(beta1, step);
    float bc2 = 1.0 - pow(beta2, step);
    float m1_hat = inertia.x / max(bc1, 1e-8);
    float m2_hat = inertia.y / max(bc2, 1e-8);

    // ── navigation_force: arrival scale (decelerate near target_loss) ─────────
    float nav_scale = navigation_force_scale();

    // ── attraction_well: compute gravity-scaled update ─────────────────────────
    float update = attraction_well(m1_hat, m2_hat, gravity_scale);

    // ── wind_field: L2 regularization push ───────────────────────────────────
    float wind = wind_field(p, 1.0);

    // ── Final parameter update ────────────────────────────────────────────────
    // Physics: p_new = p - nav_scale * (attraction_well_update + wind_decay)
    params_buf[i] = p - nav_scale * (update + wind);

    // Zero gradient after update (cleared for next backward pass)
    grads_buf[i] = 0.0;
}

// ── Gradient accumulation kernel (Win2D parallel reduce) ────────────────────

[numthreads(256, 1, 1)]
void GradAccumCS(uint3 id : SV_DispatchThreadID)
{
    // Win2D parallel: accumulate gradients from multiple mini-batch passes
    // Each thread accumulates its own parameter's gradient
    uint i = id.x;
    float g = grads_buf[i];
    // Gradient clamp per thread (attraction_well bound)
    if (isnan(g) || isinf(g)) grads_buf[i] = 0.0;
    else grads_buf[i] = clamp(g, -grad_clip, grad_clip);
}
