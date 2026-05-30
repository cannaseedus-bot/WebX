// ============================================================================
// SharedMemoryStateReader.cs - C# MMF Reader (ASX v0.7 Phase 6)
// ============================================================================

using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

namespace Micronaut.Worker.Host
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct SharedStateHeader
    {
        public uint Version;
        public uint ActiveFold;
        public uint TickCount;
        public float Entropy;
        public float Attention;
        public float Pressure;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 10)]
        public float[] Reserve;
    }

    public class SharedMemoryStateReader : IDisposable
    {
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _accessor;
        private readonly string _mapName;
        private readonly int _size;

        public SharedMemoryStateReader(string mapName = "Local\\KuhulGeometricState", int size = 4096)
        {
            _mapName = mapName;
            _size = size;

            try
            {
                // Open existing mapping
                _mmf = MemoryMappedFile.OpenExisting(_mapName);
                _accessor = _mmf.CreateViewAccessor(0, _size, MemoryMappedFileAccess.Read);
                Console.WriteLine($"[⟁] .NET: Connected to Shared Memory '{_mapName}'.");
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine($"[☢] .NET: Shared Memory '{_mapName}' not found. Is the host running?");
                throw;
            }
        }

        public SharedStateHeader ReadState()
        {
            SharedStateHeader header;
            _accessor.Read(0, out header);
            return header;
        }

        public void Dispose()
        {
            _accessor?.Dispose();
            _mmf?.Dispose();
        }
    }

    // Example Usage
    public static class SharedMemoryTest
    {
        public static void Run()
        {
            try
            {
                using var reader = new SharedMemoryStateReader();
                Console.WriteLine("Monitoring Geometric State (60fps)...");

                uint lastTick = 0;
                while (true)
                {
                    var state = reader.ReadState();
                    if (state.TickCount != lastTick)
                    {
                        Console.WriteLine($"[TICK] {state.TickCount} | FOLD {state.ActiveFold} | E={state.Entropy:F4} | A={state.Attention:F4} | P={state.Pressure:F4}");
                        lastTick = state.TickCount;
                    }
                    Thread.Sleep(16);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] {ex.Message}");
            }
        }
    }
}
