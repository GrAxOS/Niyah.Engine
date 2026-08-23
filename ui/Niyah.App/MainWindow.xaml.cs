using System;
using System.Windows;
using System.Windows.Input;
using System.Collections.ObjectModel;

namespace Niyah.App;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<SearchResult> _searchResults = new();
    private readonly ObservableCollection<string>       _docIds        = new();

    public MainWindow()
    {
        InitializeComponent();

        // Wire observable collections
        ResultsListBox.ItemsSource = _searchResults;
        DocListBox.ItemsSource     = _docIds;

        try
        {
            VersionLabel.Text   = $"v{NiyahBridge.Version}";
            DocCountLabel.Text  = $"{NiyahBridge.DocumentCount} documents";
            StatusLabel.Text    = "Ready.";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Bridge load error: {ex.Message}";
        }
    }

    // ── Search ────────────────────────────────────────────────────────────

    private void SearchTextBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) RunSearch();
    }

    private void SearchButton_Click(object sender, RoutedEventArgs e) => RunSearch();

    private void RunSearch()
    {
        var query = SearchTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(query)) return;

        try
        {
            var results = NiyahBridge.Search(query);
            _searchResults.Clear();
            foreach (var r in results) _searchResults.Add(r);

            StatusLabel.Text = results.Count > 0
                ? $"Found {results.Count} result(s) for "{query}""
                : $"No results for "{query}"";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Search error: {ex.Message}";
        }
    }

    private void ResultsListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (ResultsListBox.SelectedItem is SearchResult sr)
        {
            SelectedDocLabel.Text = $"Document: {sr.DocId}";
            var content = NiyahBridge.GetDocument(sr.DocId);
            DocPreviewBox.Text = content ?? "(content unavailable)";
        }
    }

    // ── Documents ─────────────────────────────────────────────────────────

    private void AddDocumentButton_Click(object sender, RoutedEventArgs e)
    {
        var content = DocumentTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(content)) return;

        try
        {
            string id = NiyahBridge.AddDocument(content);
            if (!string.IsNullOrEmpty(id))
            {
                _docIds.Add(id);
                DocumentTextBox.Clear();
                AddStatusLabel.Text    = $"Added: {id}";
                DocCountLabel.Text     = $"{NiyahBridge.DocumentCount} documents";
                StatusLabel.Text       = $"Document {id} added successfully.";
            }
            else
            {
                AddStatusLabel.Text = "Add failed (store full or error).";
            }
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Add error: {ex.Message}";
        }
    }

    private void DocListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (DocListBox.SelectedItem is string docId)
        {
            var content = NiyahBridge.GetDocument(docId);
            StatusLabel.Text = content != null
                ? $"Document {docId} — {content.Length} chars"
                : $"Document {docId} not found.";
        }
    }

    private void DeleteDocButton_Click(object sender, RoutedEventArgs e)
    {
        if (DocListBox.SelectedItem is not string docId) return;

        if (NiyahBridge.DeleteDocument(docId))
        {
            _docIds.Remove(docId);
            DocCountLabel.Text = $"{NiyahBridge.DocumentCount} documents";
            StatusLabel.Text   = $"Deleted {docId}.";
        }
        else
        {
            StatusLabel.Text = $"Could not delete {docId}.";
        }
    }

    // ── LLM Generation ────────────────────────────────────────────────────

    private void GenerateButton_Click(object sender, RoutedEventArgs e)
    {
        var prompt = PromptTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(prompt)) return;

        if (!int.TryParse(MaxTokensBox.Text, out int maxTokens) || maxTokens <= 0)
            maxTokens = 128;

        try
        {
            GenerateStatus.Text  = "Generating…";
            GenerateOutputBox.Clear();

            string? modelPath = string.IsNullOrWhiteSpace(ModelPathBox.Text)
                                ? null : ModelPathBox.Text.Trim();

            string output = NiyahBridge.Generate(prompt, modelPath, maxTokens);
            GenerateOutputBox.Text = output;
            GenerateStatus.Text    = $"Done — {output.Length} chars";
            StatusLabel.Text       = "Generation complete.";
        }
        catch (Exception ex)
        {
            GenerateStatus.Text = $"Error: {ex.Message}";
            StatusLabel.Text    = $"Generation error: {ex.Message}";
        }
    }
}
