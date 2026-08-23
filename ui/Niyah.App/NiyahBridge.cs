using System;
using System.Runtime.InteropServices;
using System.Text;

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

        var text = Encoding.UTF8.GetString(buffer).TrimEnd('\0');
        if (status != 0 && status != 2)
            throw new InvalidOperationException($"Native search failed: {status}");
        return text;
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
