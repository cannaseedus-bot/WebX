//================================================================================
// StabilizeCS.hlsl — GPU-accelerated logit stabilization for iGPU training
// Compile: dxc.exe -T cs_6_0 -E CSMain StabilizeCS.hlsl -Fo StabilizeCS.cso
//================================================================================

RWStructuredBuffer<float> logitsBuffer      : register(u0);
RWStructuredBuffer<float> stabilizedBuffer  : register(u1);
RWStructuredBuffer<float> gradientBuffer    : register(u2);
RWStructuredBuffer<float> clippedGradBuffer : register(u3);

static const float MAX_LOGIT = 20.0f;
static const float MAX_GRAD  =  1.0f;

[numthreads(256, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    float logit = logitsBuffer[idx];
    stabilizedBuffer[idx] = clamp(logit, -MAX_LOGIT, MAX_LOGIT);
}

[numthreads(256, 1, 1)]
void ClipGradientCS(uint3 tid : SV_DispatchThreadID)
{
    uint  idx  = tid.x;
    float grad = gradientBuffer[idx];
    clippedGradBuffer[idx] = clamp(grad, -MAX_GRAD, MAX_GRAD);
}

// Stable graph.kxml: <node phase="Sek" domain="compute" device="gpu">
//   <bind from="logits_raw" to="logits_clamped" transform="stabilize" />
// </node>
