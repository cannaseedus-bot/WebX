<!-- SEMANTIC_READER_LAW v1 -->
Semantic Reader Law: this runtime treats documents as semantic topology, not inert text. XML/TOML/CDATA/KUHUL define folds, grams, lanes, policies, geodesics, capsules, and projection targets that the reader activates lawfully.

Reader order: preserve CDATA payloads, resolve grams, enforce policy during traversal, activate folds by pressure, route geodesics, hydrate micronauts/skills, then use tensors only for ambiguity refinement.
<!-- /SEMANTIC_READER_LAW -->

# A2A Client Sample
Show how to create an A2A Client with a command line interface which invokes agents using the A2A protocol.

## Run the Sample

To run the sample, follow these steps:

1. Run the A2A client:
    ```bash
    cd A2AClient
    dotnet run
    ```  
2. Enter your request e.g. "Show me all invoices for Contoso?"

## Set Secrets with Secret Manager

The agent urls are provided as a ` ` delimited list of strings

```text
cd dotnet/samples/Demos/A2AClientServer/A2AClient

dotnet user-secrets set "A2AClient:ModelId" "..."
dotnet user-secrets set "A2AClient":ApiKey" "..."
dotnet user-secrets set "A2AClient:AgentUrls" "http://localhost:5000/policy;http://localhost:5000/invoice;http://localhost:5000/logistics"
```