<!-- SEMANTIC_READER_LAW v1 -->
Semantic Reader Law: this runtime treats documents as semantic topology, not inert text. XML/TOML/CDATA/KUHUL define folds, grams, lanes, policies, geodesics, capsules, and projection targets that the reader activates lawfully.

Reader order: preserve CDATA payloads, resolve grams, enforce policy during traversal, activate folds by pressure, route geodesics, hydrate micronauts/skills, then use tensors only for ambiguity refinement.
<!-- /SEMANTIC_READER_LAW -->
# Model Context Protocol Sample

This example demonstrates how to use Model Context Protocol tools with Semantic Kernel.

MCP is an open protocol that standardizes how applications provide context to LLMs.

For information on Model Context Protocol (MCP) please refer to the [documentation](https://modelcontextprotocol.io/introduction).

The sample shows:

1. How to connect to an MCP Server using [ModelContextProtocol](https://www.nuget.org/packages/ModelContextProtocol)
2. Retrieve the list of tools the MCP Server makes available
3. Convert the MCP tools to Semantic Kernel functions so they can be added to a Kernel instance
4. Invoke the tools from Semantic Kernel using function calling

## Installing Prerequisites

The sample requires node.js and npm to be installed. So, please install them from [here](https://nodejs.org/en/download/).
 
## Configuring Secrets or Environment Variables

The example require credentials to access OpenAI.

If you have set up those credentials as secrets within Secret Manager or through environment variables for other samples from the solution in which this project is found, they will be re-used.

### To set your secrets with Secret Manager

```text
cd dotnet/samples/Demos/ModelContextProtocolPlugin

dotnet user-secrets init

dotnet user-secrets set "OpenAI:ChatModelId" "..."
dotnet user-secrets set "OpenAI:ApiKey" "..."
 "..."
```

### To set your secrets with environment variables

Use these names:

```text
# OpenAI
OpenAI__ChatModelId
OpenAI__ApiKey
```
