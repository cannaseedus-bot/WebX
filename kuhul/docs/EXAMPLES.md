# KUHUL Code Examples

## Hello World

```kuhul
[Pop]
  [Wo X tensor<float32, [10]>]
  [Yax X]
  [Ch'en X X]
[Xul]
```

## Neural Layer (matmul + bias add)

```kuhul
[Pop]
  [Wo X tensor<float32, [1024]>]
  [Wo W tensor<float32, [1024, 512]>]
  [Wo b tensor<float32, [512]>]

  [K'ayab']
    [Yax X]
    [Yax W]
    [Sek ⊗ X W]
    [Ch'en XW X]

    [Yax XW]
    [Yax b]
    [Sek ⊕ XW b]
    [Ch'en output XW]
  [Kumk'u]
[Xul]
```

## Mesh Compression

```kuhul
[Pop]
  [Wo vertices   tensor<float32, [4096, 3]>]
  [Wo kernel     tensor<float32, [3, 1]>]
  [Wo compressed tensor<float32, [4096, 1]>]

  [K'ayab']
    [Yax vertices]
    [Yax kernel]
    [Sek ⊛ vertices kernel]
    [Ch'en compressed vertices]
  [Kumk'u]
[Xul]
```

## Physics Simulation

```kuhul
[Pop]
  [Wo position tensor<float32, [512, 3]>]
  [Wo velocity tensor<float32, [512, 3]>]
  [Wo force    tensor<float32, [512, 3]>]
  [Wo dt       tensor<float32, [1]>]

  [K'ayab']
    [Yax force]
    [Yax dt]
    [Sek ⊗ force dt]
    [Ch'en fdt force]

    [Yax velocity]
    [Yax fdt]
    [Sek ⊕ velocity fdt]
    [Ch'en velocity velocity]
  [Kumk'u]
[Xul]
```

## Point Cloud Merge

```kuhul
[Pop]
  [Wo cloudA    tensor<float32, [2048, 3]>]
  [Wo cloudB    tensor<float32, [2048, 3]>]
  [Sek ⊞ cloudA cloudB]
  [Ch'en merged cloudA]
[Xul]
```
