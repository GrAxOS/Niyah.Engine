using Microsoft.Win32;
using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Niyah.App;

public partial class MainWindow : Window
{
    private readonly NiyahBridge _engine;
    private ulong _nextDocumentId = 1;

    public MainWindow()
    {
        InitializeComponent();
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
            Filter = "Text and source files|*.txt;*.md;*.json;*.c;*.h;*.cpp;*.hpp;*.cs;*.cmake;*.sql|All files|*.*"
        };

        if (dialog.ShowDialog(this) != true) return;

        foreach (var path in dialog.FileNames)
        {
            try
            {
                var info = new FileInfo(path);
                if (info.Length == 0 || info.Length > 16L * 1024L * 1024L)
                    throw new InvalidOperationException("File is empty or exceeds the 16 MiB UI ingestion limit.");

                var text = File.ReadAllText(path, Encoding.UTF8);
                if (text.Length == 0) continue;

                var id = _nextDocumentId++;
                _engine.AddDocument(id, text, new Uri(path).AbsoluteUri, Path.GetFileName(path));
                AppendMessage("System", $"Indexed: {Path.GetFileName(path)}");
            }
            catch (Exception ex)
            {
                AppendMessage("System", $"Attachment failed: {Path.GetFileName(path)} — {ex.Message}");
            }
        }
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
