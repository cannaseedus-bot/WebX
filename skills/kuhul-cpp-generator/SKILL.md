---
name: kuhul-cpp-generator
description: Scaffold and invoke the K'UHUL to C++ generator for native and wasm targets. Use when generating code, build outputs, or template copies from K'UHUL sources.
---

# kuhul-cpp-generator

Provides scaffolding and generator invocation for the K'UHUL → C++ code generator.

Usage:
- Read tools/kuhul_cpp_generator/README.md
- Run node generate.js <output_dir> to copy templates into a build folder
- Implement transpiler extensions or invoke the generator from CI

Intents:
- generate-code
- build-native
- build-wasm
