using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace Niyah.App;

/// <summary>
/// P/Invoke wrapper for niyah.dll.
/// Struct layout must match BridgeResultItem in niyah_bridge.h exactly.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
internal struct BridgeResultItem
{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
    public string DocId;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 200)]
    public string Snippet;

    public float Score;
}

public static class NiyahBridge
{
    private const string DllName = "niyah";

    // ── Version ──────────────────────────────────────────────────────────
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr niyah_bridge_version();

    public static string Version =>
        Marshal.PtrToStringAnsi(niyah_bridge_version()) ?? "unknown";

    // ── Document management ───────────────────────────────────────────────
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    private static extern int niyah_bridge_add_document(
        string content,
        out IntPtr doc_id_out);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    private static extern int niyah_bridge_delete_document(string doc_id);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    private static extern IntPtr niyah_bridge_get_document(string doc_id);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int niyah_bridge_doc_count();

    // ── Search ────────────────────────────────────────────────────────────
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    private static extern int niyah_bridge_search(
        string query,
        out IntPtr out_results,
        out int out_count);

    // ── LLM generation ────────────────────────────────────────────────────
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl,
               CharSet = CharSet.Ansi)]
    private static extern IntPtr niyah_bridge_generate(
        string prompt,
        string? model_path,
        int max_tokens);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void niyah_bridge_free_string(IntPtr s);

    // ── Public API ────────────────────────────────────────────────────────

    public static string AddDocument(string content)
    {
        int rc = niyah_bridge_add_document(content, out IntPtr id_ptr);
        return rc == 0 ? (Marshal.PtrToStringAnsi(id_ptr) ?? "") : "";
    }

    public static bool DeleteDocument(string docId)
    {
        return niyah_bridge_delete_document(docId) == 0;
    }

    public static string? GetDocument(string docId)
    {
        var ptr = niyah_bridge_get_document(docId);
        return ptr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(ptr);
    }

    public static int DocumentCount => niyah_bridge_doc_count();

    public static List<SearchResult> Search(string query)
    {
        int rc = niyah_bridge_search(query, out IntPtr results_ptr, out int count);
        var list = new List<SearchResult>();
        if (rc != 0 || count <= 0 || results_ptr == IntPtr.Zero)
            return list;

        int stride = Marshal.SizeOf<BridgeResultItem>();
        for (int i = 0; i < count; i++)
        {
            var item = Marshal.PtrToStructure<BridgeResultItem>(
                IntPtr.Add(results_ptr, i * stride));
            list.Add(new SearchResult(item.DocId, item.Snippet, item.Score));
        }
        return list;
    }

    public static string Generate(string prompt, string? modelPath = null, int maxTokens = 128)
    {
        var ptr = niyah_bridge_generate(prompt, modelPath, maxTokens);
        if (ptr == IntPtr.Zero) return "";
        string text = Marshal.PtrToStringAnsi(ptr) ?? "";
        niyah_bridge_free_string(ptr);
        return text;
    }
}

public record SearchResult(string DocId, string Snippet, float Score);
