using System;
using System.Windows;
using System.Windows.Threading;

namespace Niyah.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Catch unhandled exceptions and show them instead of silently crashing
        DispatcherUnhandledException += OnDispatcherUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;
    }

    private static void OnDispatcherUnhandledException(object sender,
        DispatcherUnhandledExceptionEventArgs e)
    {
        MessageBox.Show(
            $"Unhandled error:\n\n{e.Exception.Message}\n\n{e.Exception.StackTrace}",
            "Niyah Engine — Error",
            MessageBoxButton.OK,
            MessageBoxImage.Error);
        e.Handled = true;
    }

    private static void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        var ex = e.ExceptionObject as Exception;
        MessageBox.Show(
            $"Fatal error:\n\n{ex?.Message ?? e.ExceptionObject?.ToString()}",
            "Niyah Engine — Fatal Error",
            MessageBoxButton.OK,
            MessageBoxImage.Error);
    }
}
