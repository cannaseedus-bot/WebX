// Minimal D3D12 compute shader: vector add
RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);

[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint i = id.x;
  Out[i] = A[i] + B[i];
}
