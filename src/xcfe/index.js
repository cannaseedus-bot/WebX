// src/xcfe/index.js
export { XCFENodeRuntime, xcfe, AT_OPCODE_MAP } from './node-runtime.js';
export { XCFEAgentRuntime, Agent, Skill, Micronaut, Tool,
         AGENT_REGISTRY, SKILL_REGISTRY, MICRONAUT_REGISTRY,
         COMMAND_REGISTRY, TOOL_REGISTRY, AGENT_OPCODE_MAP } from './agent-runtime.js';
export { SemanticLayer, OpcodeLayer, ContextLayer, Fold, HorizontalFold,
         MicroFold, FoldPipeline, registerFoldNamespaces,
         SEMANTIC_MODELS, MICRO_OPCODES } from './fold-algebra.js';
export { ProtocolLayer, JsonlStream, NotationLayer, NgramAnalyzer,
         registerCommsNamespaces, AT_STACK } from './comms-layer.js';
export { ZERO, Vigesimal, PI_CONSTANTS, PiCompute, Fibonacci, BigInteger,
         Loop, MatMul, LinAlg, Formula, MathML,
         registerMathNamespaces, MATH_OPCODE_MAP } from './math-layer.js';
export { Tick, Flux, ThreadPool, Batch, Round, Step, Mark, Mapper, Graph,
         registerTemporalNamespaces, TEMPORAL_OPCODE_MAP } from './temporal-layer.js';
export { XCFETensor, blockToTensor, TuckerDecomposition, semanticSimilarity,
         estimateRank, AT_AXES, AT_BASIS_DIMENSION,
         SEMANTIC_CLUSTERS, XCFE_TENSOR_THEOREM, RANK_11_EXAMPLE } from './tensor-algebra.js';
export { VERBS, VerbChain, EndpointRegistry, Paginator, Generator, Validator,
         Parser, Renderer, ImportManager, ClassDefinition, FunctionDef,
         ActionRunner, Program, registerImperativeNamespaces,
         IMPERATIVE_OPCODE_MAP } from './imperative-layer.js';
export { WhoContext, WhatEntity, WhereContext, WhenContext, CausalModel,
         EffectMeasure, EventBus, MutationEngine, RewardFunction, EvolutionEngine,
         CData, IOBoundary, PersistenceLayer, Publisher, SearchEngine,
         registerKnowledgeNamespaces, KNOWLEDGE_OPCODE_MAP } from './knowledge-layer.js';
export { KXMLBridge, KXMLMicronaut, FoldEntanglement, LipschitzAnalyzer,
         KXML_PHASES, PHASE_INDEX, KXML_OPCODE_TABLE, ALLOWED_TRANSITIONS,
         KXML_KUHUL_MAP, BRIDGE_THEOREMS, isValidTransition, canExecute,
         registerBridgeNamespaces } from './kxml-bridge.js';
export { GravityField, PhysicsDispatcher, GravityAdvisor, KuhulPhysicsSolver, G, antigravity, heavyGravity, negativeGravity, parseGravityAttr, registerGravityNamespaces, GRAVITY_OPCODE_MAP } from './gravity.js';

export { KernelJsonSchemaBuilder, kernelFunction, registerKernelFunction, micronautToTool, buildMicronautManifest, t } from './sk-schema-builder.js';
