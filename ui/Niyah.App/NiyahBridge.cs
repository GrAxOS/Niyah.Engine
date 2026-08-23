using System;
using System.Runtime.InteropServices;

namespace Niyah.App;

public static class NiyahBridge
{
    private const string DllName = "niyah";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr niyah_bridge_version();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int niyah_bridge_search(
        [MarshalAs(UnmanagedType.LPStr)] string query,
        out IntPtr results,
        out int count);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int niyah_bridge_add_document(
        [MarshalAs(UnmanagedType.LPStr)] string content,
        out IntPtr doc_id);

    public static string Version => Marshal.PtrToStringAnsi(niyah_bridge_version()) ?? "unknown";

    public static (string Id, float Score)[] Search(string query)
    {
        var result = niyah_bridge_search(query, out IntPtr results, out int count);
        if (result != 0 || count <= 0)
            return Array.Empty<(string, float)>();

        var resultsArray = new (string, float)[count];
        var structSize = Marshal.SizeOf<(IntPtr, float)>();

        for (int i = 0; i < count; i++)
        {
            var ptr = IntPtr.Add(results, i * structSize);
            var item = Marshal.PtrToStructure<(IntPtr, float)>(ptr);
            resultsArray[i] = (Marshal.PtrToStringAnsi(item.Item1) ?? "", item.Item2);
        }

        return resultsArray;
    }

    public static string AddDocument(string content)
    {
        var result = niyah_bridge_add_document(content, out IntPtr doc_id);
        return result == 0 ? Marshal.PtrToStringAnsi(doc_id) ?? "" : "";
    }
}
