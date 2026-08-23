using System;
using System.Windows;

namespace Niyah.App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        VersionLabel.Content = $"Niyah Engine v{NiyahBridge.Version}";
    }

    private void SearchButton_Click(object sender, RoutedEventArgs e)
    {
        var query = SearchTextBox.Text;
        if (string.IsNullOrWhiteSpace(query))
            return;

        var results = NiyahBridge.Search(query);
        ResultsListBox.Items.Clear();

        foreach (var (id, score) in results)
        {
            ResultsListBox.Items.Add($"{id} (score: {score:F2})");
        }
    }

    private void AddDocumentButton_Click(object sender, RoutedEventArgs e)
    {
        var content = DocumentTextBox.Text;
        if (string.IsNullOrWhiteSpace(content))
            return;

        var doc_id = NiyahBridge.AddDocument(content);
        StatusLabel.Content = $"Added document: {doc_id}";
    }
}
