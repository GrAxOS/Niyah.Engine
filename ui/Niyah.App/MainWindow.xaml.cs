using Microsoft.Win32;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace Niyah.App;

public partial class MainWindow : Window
{
    private readonly NiyahBridge _engine;
    private ulong _nextDocumentId = 1;

    public MainWindow()
    {
        InitializeComponent();
        AllowDrop = true;
        Drop += Window_Drop;
        DragOver += Window_DragOver;
        _engine = new NiyahBridge();
        ConversationBox.Text = $"Niyah.Engine\r\nNative: {_engine.Version}\r\nEvidence classification: explicit\r\n";
        Closed += (_, _) => _engine.Dispose();
    }

    private async void Send_Click(object sender, RoutedEventArgs e)
    {
        var query = ComposerBox.Text.Trim();
        if (query.Length == 0) return;

        AppendMessage("You", query);
        ComposerBox.Clear();
        ApplyTextDirection(query);
        SetStatus("Searching native index…");

        try
        {
            var result = await Task.Run(() => _engine.Search(query, 8));
            AppendMessage("Niyah", result.Length == 0 ? "No indexed evidence matched the query." : result);
            SetStatus("Search complete");
        }
        catch (Exception ex)
        {
            AppendMessage("Niyah", $"Native engine error: {ex.Message}");
            SetStatus("Native error");
        }
    }

    private void Attach_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "Supported text/code|*.txt;*.md;*.json;*.c;*.h;*.cpp;*.hpp;*.cc;*.cs;*.cmake;*.sql|All files|*.*"
        };

        if (dialog.ShowDialog(this) != true) return;
        foreach (var path in dialog.FileNames) IngestFile(path);
    }

    private void Window_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void Window_Drop(object sender, DragEventArgs e)
    {
        if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return;
        if (e.Data.GetData(DataFormats.FileDrop) is not string[] files) return;
        foreach (var path in files.Where(File.Exists)) IngestFile(path);
    }

    private void IngestFile(string path)
    {
        try
        {
            var info = new FileInfo(path);
            if (info.Length == 0 || info.Length > 16L * 1024L * 1024L)
                throw new InvalidOperationException("File is empty or exceeds the 16 MiB UI ingestion limit.");

            var extension = info.Extension.ToLowerInvariant();
            var supported = extension is ".txt" or ".md" or ".json" or ".c" or ".h" or ".cpp" or ".hpp" or ".cc" or ".cs" or ".cmake" or ".sql";
            if (!supported)
                throw new InvalidOperationException("The native ingestion path currently accepts UTF-8 text and source documents only.");

            var text = File.ReadAllText(path, Encoding.UTF8);
            if (text.Length == 0) return;

            var id = _nextDocumentId++;
            _engine.AddDocument(id, text, new Uri(path).AbsoluteUri, info.Name);
            ApplyTextDirection(text);
            AppendMessage("System", $"Indexed: {info.Name}");
            SetStatus("Document indexed");
        }
        catch (Exception ex)
        {
            AppendMessage("System", $"Attachment failed: {Path.GetFileName(path)} — {ex.Message}");
            SetStatus("Attachment failed");
        }
    }

    private void ApplyTextDirection(string text)
    {
        if (text.Any(IsStrongArabic))
        {
            FlowDirection = FlowDirection.RightToLeft;
            ComposerBox.FlowDirection = FlowDirection.RightToLeft;
            ConversationBox.FlowDirection = FlowDirection.RightToLeft;
        }
        else
        {
            FlowDirection = FlowDirection.LeftToRight;
            ComposerBox.FlowDirection = FlowDirection.LeftToRight;
            ConversationBox.FlowDirection = FlowDirection.LeftToRight;
        }
    }

    private static bool IsStrongArabic(char c)
    {
        return (c >= '\u0600' && c <= '\u06FF') ||
               (c >= '\u0750' && c <= '\u077F') ||
               (c >= '\u08A0' && c <= '\u08FF') ||
               (c >= '\uFB50' && c <= '\uFDFF') ||
               (c >= '\uFE70' && c <= '\uFEFF');
    }

    private void AppendMessage(string speaker, string text)
    {
        ConversationBox.AppendText($"\r\n[{speaker}]\r\n{text}\r\n");
        ConversationBox.ScrollToEnd();
    }

    private void SetStatus(string value)
    {
        if (StatusText != null) StatusText.Text = value;
    }

    protected override void OnClosed(EventArgs e)
    {
        _engine.Dispose();
        base.OnClosed(e);
    }
}
