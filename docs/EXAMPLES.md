# K'UHUL++ Code Examples

## Hello World

```kuhul
Tensor message = "Hello, KUHUL World!";
```

## Tensor Arithmetic

```kuhul
Tensor a = [1.0, 2.0, 3.0];
Tensor b = [4.0, 5.0, 6.0];
Tensor sum  = a ⊕ b;   // [5, 7, 9]
Tensor diff = a ⊖ b;   // [-3, -3, -3]
Tensor prod = a ⊗ b;   // [4, 10, 18]
```

## Dense Neural Layer (⊗ + ⊕)

```kuhul
Tensor weights = [0.5, -0.3, 0.8, -0.2, -0.1, 0.9, -0.4, 0.7, 0.3, -0.6, 0.1, 0.5];
Tensor bias    = [0.1, -0.1, 0.05, 0.0];
Tensor input   = [1.0, 0.5, -0.75];
Tensor output  = weights ⊗ input ⊕ bias;
```

## Mesh Compression (↻)

```kuhul
Tensor vertices = load_dataset("mesh_vertices");
Tensor compressed = vertices ↻ 45.0;    // rotate 45° then deduplicate
```

## Torsion Field Deformation (∿)

```kuhul
Tensor body    = load_dataset("body_verts");
Tensor twisted = body ∿ 0.2;            // 0.2 rad/unit-Y twist
```

## Radial Projection (⊙)

```kuhul
Tensor cloud      = generate_spiral(1024);
Tensor unitSphere = cloud ⊙ 1.0;        // project onto unit sphere
```

## Spherical Loop (⟲)

```kuhul
Tensor mesh   = load_dataset("mesh");
Tensor looped = mesh ⟲ 0.5π;           // spherical round-trip with phase offset
```

## Vector Encrypt (⤍)

```kuhul
Tensor verts  = load_dataset("vertices");
Tensor matrix = [1,0,0,0, 0,1,0,0, 0,0,1,0, 2,3,0,1]; // translate (2,3,0)
Tensor moved  = verts ⤍ matrix;
```

## GPU Pipeline

```kuhul
Pipeline forwardPass {
    Tensor hidden = weights1 ⊗ input  ⊕ bias1;
    Tensor output = weights2 ⊗ hidden ⊕ bias2;
}

GPU.Dispatch(forwardPass, [16, 1, 1]);
```

## Phase-Based Animation

```kuhul
Tensor phase    = 0.0;                  // will advance each frame
Tensor vertices = generate_spiral(512);
Tensor animated = vertices ⟲ phase;    // update phase each frame for animation
```

## Cluster for Model Parameters

```kuhul
Cluster layer1 {
    Tensor weights = load_dataset("layer1_w");
    Tensor bias    = load_dataset("layer1_b");
}

Tensor out = layer1.weights ⊗ input ⊕ layer1.bias;
```

---

*Example source files: `examples/kuhul/`*
*See also: [KUHUL.md](KUHUL.md), [API.md](API.md)*
