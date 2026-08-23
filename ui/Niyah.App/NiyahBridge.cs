using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Niyah.App;

internal sealed class NiyahBridge : IDisposable
{
    private const string LibraryName = "niyah_bridge";

    private IntPtr _handle;
    private bool _disposed;

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_create(out IntPtr bridge);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void niyah_bridge_destroy(IntPtr bridge);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_add_document(
        IntPtr bridge,
        ulong documentId,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? url,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? title,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string text);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_search(
        IntPtr bridge,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
        nuint limit,
        [Out] byte[] output,
        nuint outputSize);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr niyah_bridge_version();

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_generation_create(
        IntPtr bridge,
        [In] uint[]? promptTokens,
        nuint promptCount,
        nuint maximumTokens,
        out IntPtr generation);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_generation_next(
        IntPtr generation,
        out uint tokenId,
        out int finished);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void niyah_bridge_generation_cancel(IntPtr generation);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void niyah_bridge_generation_destroy(IntPtr generation);

    public NiyahBridge()
    {
        var status = niyah_bridge_create(out _handle);
        if (status != 0 || _handle == IntPtr.Zero)
            throw new InvalidOperationException($"Native engine initialization failed: {status}");
    }

    public string Version
    {
        get
        {
            EnsureNotDisposed();
            var ptr = niyah_bridge_version();
            return ptr == IntPtr.Zero ? "unknown" : Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        }
    }

    public void AddDocument(ulong documentId, string text, string? url = null, string? title = null)
    {
        EnsureNotDisposed();
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Document text is required.", nameof(text));

        var status = niyah_bridge_add_document(_handle, documentId, url, title, text);
        if (status != 0)
            throw new InvalidOperationException($"Native document indexing failed: {status}");
    }

    public string Search(string query, int limit = 8)
    {
        EnsureNotDisposed();
        if (string.IsNullOrWhiteSpace(query))
            return string.Empty;
        if (limit < 1 || limit > 1024)
            throw new ArgumentOutOfRangeException(nameof(limit));

        var buffer = new byte[64 * 1024];
        var status = niyah_bridge_search(
            _handle,
            query,
            (nuint)limit,
            buffer,
            (nuint)buffer.Length);

        var length = Array.IndexOf(buffer, (byte)0);
        if (length < 0) length = buffer.Length;
        var text = Encoding.UTF8.GetString(buffer, 0, length);
        if (status != 0 && status != 2)
            throw new InvalidOperationException($"Native search failed: {status}");
        return text;
    }

    public async IAsyncEnumerable<uint> StreamGenerationAsync(
        IReadOnlyList<uint> promptTokens,
        int maximumTokens,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureNotDisposed();
        if (maximumTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maximumTokens));
        if (promptTokens.Count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(promptTokens));

        var prompt = new uint[promptTokens.Count];
        for (var i = 0; i < prompt.Length; ++i) prompt[i] = promptTokens[i];

        var status = niyah_bridge_generation_create(
            _handle,
            prompt.Length == 0 ? null : prompt,
            (nuint)prompt.Length,
            (nuint)maximumTokens,
            out var generation);

        if (status != 0 || generation == IntPtr.Zero)
            throw new InvalidOperationException($"Native generation initialization failed: {status}");

        try
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();

                status = niyah_bridge_generation_next(generation, out var tokenId, out var finished);
                if (status == 4) throw new OperationCanceledException(cancellationToken);
                if (status == 5) throw new InvalidOperationException("Native generation weights are unavailable.");
                if (status != 0) throw new InvalidOperationException($"Native generation failed: {status}");

                if (finished != 0) yield break;
                yield return tokenId;
                await Task.Yield();
            }
        }
        finally
        {
            niyah_bridge_generation_cancel(generation);
            niyah_bridge_generation_destroy(generation);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            niyah_bridge_destroy(_handle);
            _handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    ~NiyahBridge() => Dispose();

    private void EnsureNotDisposed()
    {
        if (_disposed || _handle == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(NiyahBridge));
    }
}
