#!/usr/bin/env python3
"""
Native system dialog utilities for cross-platform UI.

macOS: shells out to osascript so we get genuine Aqua sheets.
Windows / Linux: uses PyQt5's QMessageBox / QInputDialog. The app's
QApplication is already alive by the time any of these are called
(see main.py:437), so the dialog plugs into the existing Qt event loop
instead of spawning a separate Tk top-level. Tk on Windows comes up as
a full-decorated top-level window that fights with our Qt main window
for modal focus and feels nothing like a system dialog -- avoid it.
A Tk path is preserved at the bottom of each function purely as a
last-resort fallback if Qt isn't importable for some reason.
"""

import sys
import subprocess
import platform


def _is_macos():
    return platform.system() == 'Darwin'


def _qt_app():
    """Return the running QApplication or None. Importing PyQt5 here
    (not at module top) so this module stays importable in CLI tools
    that don't pull in Qt."""
    try:
        from PyQt5.QtWidgets import QApplication
        return QApplication.instance()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# show_info / show_warning / show_error
# ---------------------------------------------------------------------------

def _qt_messagebox(icon_name: str, title: str, message: str) -> bool:
    """Try to show a Qt QMessageBox. Returns True on success."""
    try:
        from PyQt5.QtWidgets import QMessageBox
        if _qt_app() is None:
            return False
        box = QMessageBox()
        box.setIcon(getattr(QMessageBox, icon_name, QMessageBox.Information))
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec_()
        return True
    except Exception:
        return False


def show_info(title, message):
    if _is_macos():
        try:
            script = f'display dialog "{message}" with title "{title}" buttons {{"OK"}} default button "OK" with icon note'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")
    if _qt_messagebox('Information', title, message):
        return
    # Last-resort Tk fallback
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


def show_warning(title, message):
    if _is_macos():
        try:
            script = f'display dialog "{message}" with title "{title}" buttons {{"OK"}} default button "OK" with icon caution'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")
    if _qt_messagebox('Warning', title, message):
        return
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showwarning(title, message)
        root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


def show_error(title, message):
    if _is_macos():
        try:
            script = f'display dialog "{message}" with title "{title}" buttons {{"OK"}} default button "OK" with icon stop'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")
    if _qt_messagebox('Critical', title, message):
        return
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


# ---------------------------------------------------------------------------
# ask_yes_no
# ---------------------------------------------------------------------------

def ask_yes_no(title, message, default_yes=True):
    if _is_macos():
        try:
            default_btn = "Yes" if default_yes else "No"
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"No", "Yes"}} default button "{default_btn}" with icon note'
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            return "Yes" in result.stdout
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")
            import traceback
            traceback.print_exc()

    # Qt path
    try:
        from PyQt5.QtWidgets import QMessageBox
        if _qt_app() is not None:
            box = QMessageBox()
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(message)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.Yes if default_yes else QMessageBox.No)
            return box.exec_() == QMessageBox.Yes
    except Exception:
        pass

    # Tk last-resort
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        result = messagebox.askyesno(title, message, default='yes' if default_yes else 'no')
        root.destroy()
        return result
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        return False


# ---------------------------------------------------------------------------
# ask_three_choice
# ---------------------------------------------------------------------------

def ask_three_choice(title, message, button1, button2, button3, default_button=2):
    """Returns 1, 2, or 3 (or None if cancelled)."""
    if _is_macos():
        try:
            default_btn_text = [button1, button2, button3][default_button - 1]
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"{button3}", "{button2}", "{button1}"}} default button "{default_btn_text}" with icon note'
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if f"button returned:{button1}" in result.stdout:
                return 1
            elif f"button returned:{button2}" in result.stdout:
                return 2
            elif f"button returned:{button3}" in result.stdout:
                return 3
            return None
        except subprocess.CalledProcessError:
            return None
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")

    # Qt path: QMessageBox with three custom buttons
    try:
        from PyQt5.QtWidgets import QMessageBox
        if _qt_app() is not None:
            box = QMessageBox()
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle(title)
            box.setText(message)
            b1 = box.addButton(button1, QMessageBox.AcceptRole)
            b2 = box.addButton(button2, QMessageBox.AcceptRole)
            b3 = box.addButton(button3, QMessageBox.RejectRole)
            default_btn = [b1, b2, b3][default_button - 1]
            box.setDefaultButton(default_btn)
            box.exec_()
            clicked = box.clickedButton()
            if clicked is b1: return 1
            if clicked is b2: return 2
            if clicked is b3: return 3
            return None
    except Exception:
        pass

    # Tk last-resort (the original three-button custom dialog)
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.resizable(False, False)
        msg_label = tk.Label(dialog, text=message, padx=20, pady=20, wraplength=400)
        msg_label.pack()
        btn_frame = tk.Frame(dialog, padx=10, pady=10)
        btn_frame.pack()
        result = [None]
        def on_button(choice):
            result[0] = choice; dialog.destroy(); root.destroy()
        btn1 = tk.Button(btn_frame, text=button1, command=lambda: on_button(1), width=15); btn1.pack(side=tk.LEFT, padx=5)
        btn2 = tk.Button(btn_frame, text=button2, command=lambda: on_button(2), width=15); btn2.pack(side=tk.LEFT, padx=5)
        btn3 = tk.Button(btn_frame, text=button3, command=lambda: on_button(3), width=15); btn3.pack(side=tk.LEFT, padx=5)
        if default_button == 1: btn1.focus_set()
        elif default_button == 2: btn2.focus_set()
        else: btn3.focus_set()
        dialog.update_idletasks()
        width = dialog.winfo_width(); height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        dialog.transient(root); dialog.grab_set()
        root.wait_window(dialog)
        return result[0]
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        return None


# Test functions
if __name__ == "__main__":
    print("Testing native dialogs...")
    show_info("Test Info", "This is an info message")
    show_warning("Test Warning", "This is a warning message")
    show_error("Test Error", "This is an error message")
    result = ask_yes_no("Test Question", "Do you want to continue?")
    print(f"   Result: {result}")
    result = ask_three_choice("Test Three Choice", "What would you like to do?",
                              "Option 1", "Option 2", "Option 3", default_button=2)
    print(f"   Result: {result}")
    print("\nAll tests complete!")
