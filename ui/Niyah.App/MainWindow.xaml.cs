using System.Windows;

namespace Niyah.App;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        ConversationBox.Text =
            "Niyah.Engine\r\n\r\n" +
            "State: local shell\r\n" +
            "Evidence: explicit\r\n" +
            "Emotion simulation: disabled\r\n" +
            "External provider dependency: none\r\n";
    }

    private void Send_Click(object sender, RoutedEventArgs e)
    {
        var text = ComposerBox.Text.Trim();
        if (text.Length == 0)
            return;

        ConversationBox.AppendText($"\r\nYou: {text}\r\n");
        ConversationBox.AppendText("Niyah: input accepted. Reasoning and source retrieval are not connected yet.\r\n");
        ComposerBox.Clear();
    }

    private static void Tab_Click(object sender, RoutedEventArgs e)
    {
        // UI shell only. Feature views will be connected to real storage/engine APIs.
    }
}
