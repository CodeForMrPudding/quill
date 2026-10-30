import threading
from typing import Optional


class ToastNotifier:
    """
    Thin wrapper around windows-toasts' InteractableWindowsToaster. Imported
    lazily (only when actually constructed) so the rest of the app doesn't
    require this Windows-only dependency just to import cli.py.
    """

    def __init__(self, app_name: str = "Quill"):
        from windows_toasts import InteractableWindowsToaster
        self._toaster = InteractableWindowsToaster(app_name)

    def notify(self, title: str, message: str) -> None:
        from windows_toasts import Toast
        self._toaster.show_toast(Toast([title, message]))

    def ask_yes_no(self, title: str, message: str, timeout: float = 300.0) -> Optional[bool]:
        """
        Shows a toast with Yes/No buttons and blocks (up to `timeout`
        seconds) until the user responds. Returns True for an explicit "Yes",
        False for an explicit "No", and None when there was no answer
        (dismissed, ignored, timed out) -- silence is never consent to write
        into the vault, but it isn't a "No" either, so the caller can ask
        again later instead of treating the PDF as declined.
        """
        from windows_toasts import (
            Toast,
            ToastButton,
            ToastDuration,
            ToastScenario,
            ToastActivatedEventArgs,
            ToastDismissedEventArgs,
        )

        decision = {"value": None}
        event = threading.Event()

        toast = Toast([title, message])
        toast.AddAction(ToastButton("Yes", "yes"))
        toast.AddAction(ToastButton("No", "no"))
        # Reminder scenario keeps the toast on screen until the user acts on
        # it, instead of the default ~5s auto-dismiss into the action center
        # (confirmed via live testing: the default duration is too short for
        # a human to notice and click before it times out).
        toast.duration = ToastDuration.Long
        toast.scenario = ToastScenario.Reminder

        def on_activated(args: ToastActivatedEventArgs):
            decision["value"] = args.arguments == "yes"
            event.set()

        def on_dismissed(args: ToastDismissedEventArgs):
            event.set()

        toast.on_activated = on_activated
        toast.on_dismissed = on_dismissed

        self._toaster.show_toast(toast)
        event.wait(timeout=timeout)
        return decision["value"]


class ConsoleNotifier:
    """
    Fallback when Windows toasts aren't available (other platforms, or
    windows-toasts not installed). It can't ask interactively from a
    background thread, so it answers with `auto_approve`: True drafts
    every new PDF; None (the default) leaves it for later -- run
    `quill watch --auto-approve`, or use /draft-all in chat.
    """

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve

    def notify(self, title: str, message: str) -> None:
        print(f"[{title}] {message}")

    def ask_yes_no(self, title: str, message: str, timeout: float = 0) -> Optional[bool]:
        print(f"[{title}] {message} -> {'yes (--auto-approve)' if self.auto_approve else 'skipped (no way to ask; use --auto-approve)'}")
        return True if self.auto_approve else None
