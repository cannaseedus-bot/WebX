// PHASE 3.1: K'UHUL SEMANTIC SKILL INTEGRATION WITH SEMANTIC KERNEL
// 
// This C# bridge connects the K'UHUL semantic skill system (Node.js/JavaScript)
// to the Microsoft Semantic Kernel (.NET/C#) via JSON-RPC and interop calls.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Functions;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.TextGeneration;

namespace SemanticKernel.Plugins.KuhulIntegration
{
    /// <summary>
    /// K'UHUL Semantic Skill Plugin for Semantic Kernel
    /// 
    /// Bridges 62+ K'UHUL semantic skills into SK function registry
    /// via JSON-RPC communication with the Node.js semantic skill executor.
    /// </summary>
    public class KuhulSemanticSkillPlugin
    {
        private readonly HttpClient _httpClient;
        private readonly string _skillExecutorUrl;
        private readonly Dictionary<string, KuhulSkillMetadata> _skillCache;
        private readonly int _rpcIdCounter;

        public KuhulSemanticSkillPlugin(string skillExecutorUrl = "http://127.0.0.1:25100")
        {
            _httpClient = new HttpClient();
            _skillExecutorUrl = skillExecutorUrl;
            _skillCache = new Dictionary<string, KuhulSkillMetadata>();
            _rpcIdCounter = 0;
        }

        /// <summary>
        /// Register all K'UHUL skills as SK functions
        /// </summary>
        public async Task<int> RegisterSkillsAsync(Kernel kernel)
        {
            try
            {
                // Discover all available skills from K'UHUL registry
                var skills = await DiscoverSkillsAsync();

                int registered = 0;

                foreach (var skill in skills)
                {
                    try
                    {
                        // Register each skill as an SK function
                        RegisterSkillAsFunction(kernel, skill);
                        _skillCache[skill.Id] = skill;
                        registered++;
                    }
                    catch (Exception ex)
                    {
                        System.Console.WriteLine($"Warning: Failed to register skill {skill.Capability}: {ex.Message}");
                    }
                }

                return registered;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Failed to register K'UHUL skills", ex);
            }
        }

        /// <summary>
        /// Discover available skills from K'UHUL system
        /// </summary>
        private async Task<List<KuhulSkillMetadata>> DiscoverSkillsAsync()
        {
            var rpcRequest = new JsonRpcRequest
            {
                JsonRpc = "2.0",
                Method = "skill/registry",
                Id = NextRpcId()
            };

            var response = await InvokeRpcAsync(rpcRequest);

            if (response.Error != null)
            {
                throw new InvalidOperationException($"Failed to discover skills: {response.Error.Message}");
            }

            var skillsList = new List<KuhulSkillMetadata>();

            if (response.Result is JsonElement resultElement && resultElement.ValueKind == JsonValueKind.Object)
            {
                var domainProperty = resultElement.GetProperty("skills_by_domain");
                
                foreach (var domain in domainProperty.EnumerateObject())
                {
                    if (domain.Value.ValueKind == JsonValueKind.Number)
                    {
                        var count = domain.Value.GetInt32();
                        // Skills will be fetched per-domain
                    }
                }

                // Fetch detailed skill list
                var discoverRequest = new JsonRpcRequest
                {
                    JsonRpc = "2.0",
                    Method = "skill/discover",
                    Params = new { min_confidence = 0.75 },
                    Id = NextRpcId()
                };

                var discoverResponse = await InvokeRpcAsync(discoverRequest);

                if (discoverResponse.Result is JsonElement skillsElement && 
                    skillsElement.TryGetProperty("skills", out var skillsArray))
                {
                    foreach (var skillJson in skillsArray.EnumerateArray())
                    {
                        var skill = ParseSkillMetadata(skillJson);
                        skillsList.Add(skill);
                    }
                }
            }

            return skillsList;
        }

        /// <summary>
        /// Register a single K'UHUL skill as an SK native function
        /// </summary>
        private void RegisterSkillAsFunction(Kernel kernel, KuhulSkillMetadata skill)
        {
            // Create a native function that calls the K'UHUL skill executor
            var function = KernelFunctionFactory.CreateFromMethod(
                method: async (string input) => await ExecuteKuhulSkillAsync(skill.Id, input),
                functionName: skill.Capability.Replace(".", "_").ToLower(),
                description: skill.Description,
                parameters: new[]
                {
                    new KernelParameterMetadata
                    {
                        Name = "input",
                        Description = "Natural language query or command",
                        DefaultValue = ""
                    }
                }
            );

            // Register in plugin collection
            kernel.Plugins.AddFromFunctions(
                $"kuhul_{skill.Domain}",
                new[] { (skill.Capability, function) }
            );
        }

        /// <summary>
        /// Execute a K'UHUL skill via JSON-RPC
        /// </summary>
        private async Task<string> ExecuteKuhulSkillAsync(string skillId, string query)
        {
            var rpcRequest = new JsonRpcRequest
            {
                JsonRpc = "2.0",
                Method = "skill/semantic",
                Params = new { query, skill_id = skillId },
                Id = NextRpcId()
            };

            try
            {
                var response = await InvokeRpcAsync(rpcRequest);

                if (response.Error != null)
                {
                    return $"Error: {response.Error.Message}";
                }

                if (response.Result is JsonElement resultElement &&
                    resultElement.TryGetProperty("result", out var result))
                {
                    return result.ToString();
                }

                return response.Result?.ToString() ?? "No result";
            }
            catch (Exception ex)
            {
                return $"Execution error: {ex.Message}";
            }
        }

        /// <summary>
        /// Invoke K'UHUL JSON-RPC endpoint
        /// </summary>
        private async Task<JsonRpcResponse> InvokeRpcAsync(JsonRpcRequest request)
        {
            var jsonContent = JsonSerializer.Serialize(request);
            var httpContent = new StringContent(jsonContent, System.Text.Encoding.UTF8, "application/json");

            try
            {
                var httpResponse = await _httpClient.PostAsync(
                    $"{_skillExecutorUrl}/rpc",
                    httpContent
                );

                httpResponse.EnsureSuccessStatusCode();

                var responseJson = await httpResponse.Content.ReadAsStringAsync();
                var response = JsonSerializer.Deserialize<JsonRpcResponse>(responseJson);

                return response ?? throw new InvalidOperationException("Invalid RPC response");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"RPC call failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Parse K'UHUL skill metadata from JSON
        /// </summary>
        private KuhulSkillMetadata ParseSkillMetadata(JsonElement skillJson)
        {
            return new KuhulSkillMetadata
            {
                Id = skillJson.GetProperty("id").GetString() ?? "",
                Capability = skillJson.GetProperty("capability").GetString() ?? "",
                Domain = skillJson.GetProperty("domain").GetString() ?? "",
                Description = skillJson.GetProperty("description").GetString() ?? "",
                Confidence = skillJson.GetProperty("confidence").GetDouble(),
                LatencyMs = skillJson.GetProperty("latency_ms").GetInt32(),
                Supports = skillJson.GetProperty("supports")
                    .EnumerateArray()
                    .Select(x => x.GetString() ?? "")
                    .ToList()
            };
        }

        private int NextRpcId() => ++_rpcIdCounter;

        /// <summary>
        /// Get skill execution statistics
        /// </summary>
        public async Task<KuhulExecutionStats> GetStatisticsAsync()
        {
            var rpcRequest = new JsonRpcRequest
            {
                JsonRpc = "2.0",
                Method = "skill/registry",
                Id = NextRpcId()
            };

            var response = await InvokeRpcAsync(rpcRequest);

            if (response.Result is JsonElement resultElement)
            {
                return new KuhulExecutionStats
                {
                    TotalSkills = resultElement.GetProperty("total_skills").GetInt32(),
                    TotalDomains = resultElement.GetProperty("total_domains").GetInt32(),
                    AverageConfidence = resultElement.GetProperty("average_confidence").GetDouble(),
                    AverageLatencyMs = resultElement.GetProperty("average_latency_ms").GetInt32()
                };
            }

            throw new InvalidOperationException("Failed to get statistics");
        }
    }

    /// <summary>
    /// K'UHUL skill metadata
    /// </summary>
    public class KuhulSkillMetadata
    {
        public string Id { get; set; } = "";
        public string Capability { get; set; } = "";
        public string Domain { get; set; } = "";
        public string Description { get; set; } = "";
        public double Confidence { get; set; }
        public int LatencyMs { get; set; }
        public List<string> Supports { get; set; } = new();
    }

    /// <summary>
    /// K'UHUL execution statistics
    /// </summary>
    public class KuhulExecutionStats
    {
        public int TotalSkills { get; set; }
        public int TotalDomains { get; set; }
        public double AverageConfidence { get; set; }
        public int AverageLatencyMs { get; set; }
    }

    /// <summary>
    /// JSON-RPC 2.0 request envelope
    /// </summary>
    public class JsonRpcRequest
    {
        [System.Text.Json.Serialization.JsonPropertyName("jsonrpc")]
        public string JsonRpc { get; set; } = "2.0";

        [System.Text.Json.Serialization.JsonPropertyName("method")]
        public string Method { get; set; } = "";

        [System.Text.Json.Serialization.JsonPropertyName("params")]
        public object? Params { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("id")]
        public int Id { get; set; }
    }

    /// <summary>
    /// JSON-RPC 2.0 response envelope
    /// </summary>
    public class JsonRpcResponse
    {
        [System.Text.Json.Serialization.JsonPropertyName("jsonrpc")]
        public string JsonRpc { get; set; } = "2.0";

        [System.Text.Json.Serialization.JsonPropertyName("result")]
        public object? Result { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("error")]
        public JsonRpcError? Error { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("id")]
        public int Id { get; set; }
    }

    /// <summary>
    /// JSON-RPC error response
    /// </summary>
    public class JsonRpcError
    {
        [System.Text.Json.Serialization.JsonPropertyName("code")]
        public int Code { get; set; }

        [System.Text.Json.Serialization.JsonPropertyName("message")]
        public string Message { get; set; } = "";

        [System.Text.Json.Serialization.JsonPropertyName("data")]
        public object? Data { get; set; }
    }

    /// <summary>
    /// Extension methods for Kernel to easily add K'UHUL skills
    /// </summary>
    public static class KuhulSkillExtensions
    {
        /// <summary>
        /// Add all K'UHUL semantic skills to kernel
        /// </summary>
        public static async Task<Kernel> AddKuhulSkillsAsync(
            this Kernel kernel,
            string skillExecutorUrl = "http://127.0.0.1:25100")
        {
            var plugin = new KuhulSemanticSkillPlugin(skillExecutorUrl);
            int registered = await plugin.RegisterSkillsAsync(kernel);

            System.Console.WriteLine($"✓ Registered {registered} K'UHUL semantic skills");

            // Print statistics
            try
            {
                var stats = await plugin.GetStatisticsAsync();
                System.Console.WriteLine($"  Total skills available: {stats.TotalSkills}");
                System.Console.WriteLine($"  Total domains: {stats.TotalDomains}");
                System.Console.WriteLine($"  Average confidence: {stats.AverageConfidence:P}");
                System.Console.WriteLine($"  Average latency: {stats.AverageLatencyMs}ms");
            }
            catch (Exception ex)
            {
                System.Console.WriteLine($"Warning: Could not retrieve statistics: {ex.Message}");
            }

            return kernel;
        }
    }
}
