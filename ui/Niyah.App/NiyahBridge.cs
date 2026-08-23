using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Niyah.App
{
    public enum NiyahBridgeStatus : uint
    {
        OK = 0,
        Error = 1,
        OutOfMemory = 2,
        InvalidArgs = 3,
        Cancelled = 4,
        Unavailable = 5
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct NiyahLlmConfig
    {
        public uint VocabSize;
        public uint ContextSize;
        public uint Dim;
        public uint HiddenDim;
        public uint LayerCount;
        public uint Heads;
        public uint KvHeads;
        public uint EosToken;
    }

    public class NiyahBridge : IDisposable
    {
        private IntPtr _handle = IntPtr.Zero;
        private bool _disposed = false;

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_create(out IntPtr handle);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern void niyah_bridge_destroy(IntPtr handle);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_model_validate(
            IntPtr handle, ref NiyahLlmConfig config);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_weights_load(
            IntPtr handle, byte[] buffer, UIntPtr buffer_size);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_add_document(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string id,
            [MarshalAs(UnmanagedType.LPStr)] string url,
            [MarshalAs(UnmanagedType.LPStr)] string title,
            [MarshalAs(UnmanagedType.LPStr)] string text);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_search(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string query,
            out IntPtr results,
            out UIntPtr result_count,
            UIntPtr max_results);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_generation_create(
            out IntPtr gen_handle,
            IntPtr bridge_handle,
            uint[] prompt_tokens,
            UIntPtr prompt_count,
            UIntPtr maximum_tokens);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_generation_next(
            IntPtr gen_handle,
            out uint token_id,
            out float probability,
            out bool finished);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern NiyahBridgeStatus niyah_bridge_generation_cancel(
            IntPtr gen_handle);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern void niyah_bridge_generation_destroy(
            IntPtr gen_handle);

        [DllImport("niyah_bridge", CallingConvention = CallingConvention.Cdecl)]
        private static extern void niyah_bridge_free_results(
            IntPtr results);

        public NiyahBridge()
        {
            var status = niyah_bridge_create(out _handle);
            if (status != NiyahBridgeStatus.OK)
            {
                throw new InvalidOperationException($"Failed to create bridge: {status}");
            }
        }

        public void ValidateModel(NiyahLlmConfig config)
        {
            var status = niyah_bridge_model_validate(_handle, ref config);
            if (status != NiyahBridgeStatus.OK)
            {
                throw new InvalidOperationException($"Model validation failed: {status}");
            }
        }

        public void LoadWeights(byte[] buffer)
        {
            var status = niyah_bridge_weights_load(_handle, buffer, new UIntPtr((uint)buffer.Length));
            if (status != NiyahBridgeStatus.OK)
            {
                throw new InvalidOperationException($"Weight loading failed: {status}");
            }
        }

        public void AddDocument(string id, string url, string title, string text)
        {
            var status = niyah_bridge_add_document(_handle, id, url, title, text);
            if (status != NiyahBridgeStatus.OK)
            {
                throw new InvalidOperationException($"Failed to add document: {status}");
            }
        }

        public List<(string Id, float Score)> Search(string query, uint maxResults = 10)
        {
            var status = niyah_bridge_search(_handle, query, out IntPtr results, out UIntPtr count, new UIntPtr(maxResults));
            if (status != NiyahBridgeStatus.OK || results == IntPtr.Zero)
            {
                return new List<(string, float)>();
            }

            var resultList = new List<(string, float)>();
            string[] lines = Marshal.PtrToStringAnsi(results).Split('\n');
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split('\t');
                if (parts.Length >= 2 && float.TryParse(parts[1], out float score))
                {
                    resultList.Add((parts[0], score));
                }
            }

            niyah_bridge_free_results(results);
            return resultList;
        }

        public async IAsyncEnumerable<uint> StreamGenerationAsync(
            uint[] promptTokens, uint maxTokens = 256)
        {
            var status = niyah_bridge_generation_create(
                out IntPtr genHandle, _handle, promptTokens, 
                new UIntPtr((uint)promptTokens.Length), new UIntPtr(maxTokens));

            if (status != NiyahBridgeStatus.OK)
            {
                throw new InvalidOperationException($"Generation creation failed: {status}");
            }

            try
            {
                while (true)
                {
                    await Task.Delay(1); // Yield to UI thread

                    status = niyah_bridge_generation_next(
                        genHandle, out uint token, out float prob, out bool finished);

                    if (status == NiyahBridgeStatus.Cancelled || finished)
                    {
                        break;
                    }

                    if (status == NiyahBridgeStatus.OK)
                    {
                        yield return token;
                    }
                    else if (status == NiyahBridgeStatus.Unavailable)
                    {
                        // Native generation weights are unavailable
                        // Fall back to echoing prompt tokens
                        foreach (var t in promptTokens)
                        {
                            yield return t;
                        }
                        break;
                    }
                    else
                    {
                        throw new InvalidOperationException($"Generation failed: {status}");
                    }
                }
            }
            finally
            {
                niyah_bridge_generation_destroy(genHandle);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    niyah_bridge_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}
