// ============================================================================
// SyncWorker.cs - C# Synchronized Worker (ASX v0.7 Phase 6)
// ============================================================================

using System;
using System.Threading;
using System.Threading.Tasks;

namespace Micronaut.Worker.Host
{
    public class SyncWorker
    {
        private readonly SharedMemoryStateReader _stateReader;
        private bool _running = true;

        public SyncWorker()
        {
            _stateReader = new SharedMemoryStateReader();
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            Console.WriteLine("[⟁] .NET SyncWorker: Starting. Waiting for geometric ticks...");

            uint lastTick = 0;

            try
            {
                while (!cancellationToken.IsCancellationRequested && _running)
                {
                    var state = _stateReader.ReadState();

                    // ⟁ LAW B/D Synchronized Execution
                    // The worker only executes when the global tick increments
                    if (state.TickCount != lastTick)
                    {
                        Console.WriteLine($"[⟁] .NET Sync: Executing task for Tick {state.TickCount} (Fold {state.ActiveFold})");
                        
                        // Simulate specialized C# reasoning/compute
                        // In a real system, this would call a Semantic Kernel function
                        await PerformSpecializedTask(state);

                        lastTick = state.TickCount;
                    }

                    // Poll every 1ms for high precision
                    await Task.Delay(1, cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("[⟁] .NET SyncWorker: Stopping...");
            }
            finally
            {
                _stateReader.Dispose();
            }
        }

        private async Task PerformSpecializedTask(SharedStateHeader state)
        {
            // Simulate work based on manifold state
            if (state.Entropy > 0.5f)
            {
                Console.WriteLine("    [!] High Entropy detected. Stabilizing manifold...");
            }
            
            if (state.Attention > 0.8f)
            {
                Console.WriteLine("    [*] High Attention focused on specialized task.");
            }

            // Simulate compute time
            await Task.Delay(10); 
        }

        public static async Task Main(string[] args)
        {
            var worker = new SyncWorker();
            var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            await worker.StartAsync(cts.Token);
        }
    }
}
