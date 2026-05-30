<!-- SEMANTIC_READER_LAW v1 -->
Semantic Reader Law: this runtime treats documents as semantic topology, not inert text. XML/TOML/CDATA/KUHUL define folds, grams, lanes, policies, geodesics, capsules, and projection targets that the reader activates lawfully.

Reader order: preserve CDATA payloads, resolve grams, enforce policy during traversal, activate folds by pressure, route geodesics, hydrate micronauts/skills, then use tensors only for ambiguity refinement.
<!-- /SEMANTIC_READER_LAW -->
# Microsoft.SemanticKernel.Connectors.Milvus

This is an implementation of the Semantic Kernel Memory Store abstraction for the [Milvus vector database](https://milvus.io).

**Note:** Currently, only Milvus v2.2 is supported. v2.3 is coming soon, older versions are untested.

## Quickstart using a standalone Milvus installation

1. Download the Milvus docker-compose.yml:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.2.14/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

2. Start Milvus:

```bash
docker-compose up -d
```

3. Use Semantic Kernel with Milvus, connecting to `localhost` with the default (gRPC) port of 1536:

```csharp
using MilvusMemoryStore memoryStore = new("localhost");

var embeddingGenerator = new OpenAITextEmbeddingGenerationService("text-embedding-ada-002", apiKey);

SemanticTextMemory textMemory = new(memoryStore, embeddingGenerator);

var memoryPlugin = kernel.ImportPluginFromObject(new TextMemoryPlugin(textMemory));
```

More information on setting up Milvus can be found [here](https://milvus.io/docs/v2.2.x/install_standalone-docker.md). The `MilvusMemoryStore` constructor provides additional configuration options, such as the vector size, the similarity metric type, etc.
