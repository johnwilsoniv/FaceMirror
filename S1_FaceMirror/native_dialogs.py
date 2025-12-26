#!/usr/bin/env python3
"""
Native system dialog utilities for cross-platform UI

Provides native OS dialogs instead of Qt dialogs for better system integration.
Uses osascript on macOS and tkinter as fallback for other platforms.
"""

import sys
import subprocess
import platform


def _is_macos():
    """Check if running on macOS"""
    return platform.system() == 'Darwin'


def show_info(title, message):
    """
    Show native info dialog

    Args:
        title: Dialog title
        message: Dialog message
    """
    if _is_macos():
        try:
            # Escape quotes and backslashes in message for AppleScript
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"OK"}} default button "OK" with icon note'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")

    # Fallback to tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        # Reuse existing root if available to prevent multiple app instances
        root_created = False
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            root_created = True
        messagebox.showinfo(title, message)
        if root_created:
            root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


def show_warning(title, message):
    """
    Show native warning dialog

    Args:
        title: Dialog title
        message: Dialog message
    """
    if _is_macos():
        try:
            # Escape quotes and backslashes in message for AppleScript
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"OK"}} default button "OK" with icon caution'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")

    # Fallback to tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        # Reuse existing root if available to prevent multiple app instances
        root_created = False
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            root_created = True
        messagebox.showwarning(title, message)
        if root_created:
            root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


def show_error(title, message):
    """
    Show native error dialog

    Args:
        title: Dialog title
        message: Dialog message
    """
    if _is_macos():
        try:
            # Escape quotes and backslashes in message for AppleScript
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"OK"}} default button "OK" with icon stop'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")

    # Fallback to tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        # Reuse existing root if available to prevent multiple app instances
        root_created = False
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            root_created = True
        messagebox.showerror(title, message)
        if root_created:
            root.destroy()
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        print(f"{title}: {message}")


def ask_yes_no(title, message, default_yes=True):
    """
    Show native yes/no question dialog

    Args:
        title: Dialog title
        message: Dialog message
        default_yes: Whether "Yes" is the default button

    Returns:
        True if Yes, False if No
    """
    if _is_macos():
        try:
            default_btn = "Yes" if default_yes else "No"
            # Escape quotes and backslashes in message for AppleScript
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            # Use icon note (1) for question dialogs - AppleScript doesn't have a "question" icon constant
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"No", "Yes"}} default button "{default_btn}" with icon note'
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)

            # Debug output
            print(f"DEBUG native_dialogs: osascript return code: {result.returncode}")
            print(f"DEBUG native_dialogs: stdout: '{result.stdout}'")
            print(f"DEBUG native_dialogs: stderr: '{result.stderr}'")

            # Check return code - 0 means success, non-zero means cancelled or error
            if result.returncode != 0:
                print(f"DEBUG native_dialogs: User cancelled or error (return code {result.returncode})")
                return False

            # osascript returns "button returned:Yes" or "button returned:No"
            is_yes = "Yes" in result.stdout
            print(f"DEBUG native_dialogs: Parsed result as Yes={is_yes}")
            return is_yes
        except subprocess.CalledProcessError:
            # User cancelled or error
            print(f"DEBUG native_dialogs: CalledProcessError caught")
            return False
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")
            import traceback
            traceback.print_exc()

    # Fallback to tkinter
    try:
        import tkinter as tk
        from tkinter import messagebox
        # Reuse existing root if available to prevent multiple app instances
        root_created = False
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            root_created = True
        result = messagebox.askyesno(title, message, default='yes' if default_yes else 'no')
        if root_created:
            root.destroy()
        return result
    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        return False


def ask_three_choice(title, message, button1, button2, button3, default_button=2):
    """
    Show native dialog with three choice buttons

    Args:
        title: Dialog title
        message: Dialog message
        button1: Text for first button (leftmost)
        button2: Text for second button (middle)
        button3: Text for third button (rightmost)
        default_button: Which button is default (1, 2, or 3)

    Returns:
        1, 2, or 3 corresponding to which button was clicked, or None if cancelled
    """
    if _is_macos():
        try:
            default_btn_text = [button1, button2, button3][default_button - 1]
            # Escape quotes and backslashes in message for AppleScript
            message_escaped = message.replace('\\', '\\\\').replace('"', '\\"')
            title_escaped = title.replace('\\', '\\\\').replace('"', '\\"')
            # In macOS, buttons appear right-to-left, so we reverse them
            # Use icon note for question dialogs
            script = f'display dialog "{message_escaped}" with title "{title_escaped}" buttons {{"{button3}", "{button2}", "{button1}"}} default button "{default_btn_text}" with icon note'
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)

            # Parse which button was clicked
            if f"button returned:{button1}" in result.stdout:
                return 1
            elif f"button returned:{button2}" in result.stdout:
                return 2
            elif f"button returned:{button3}" in result.stdout:
                return 3
            return None
        except subprocess.CalledProcessError:
            # User cancelled
            return None
        except Exception as e:
            print(f"Warning: Failed to show native dialog: {e}")

    # Fallback to tkinter (limited to 3 buttons using custom dialog)
    try:
        import tkinter as tk
        from tkinter import messagebox

        # Reuse existing root if available to prevent multiple app instances
        root_created = False
        if tk._default_root is None:
            root = tk.Tk()
            root.withdraw()
            root_created = True
        else:
            root = tk._default_root

        # Create custom dialog window
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.resizable(False, False)

        # Message label
        msg_label = tk.Label(dialog, text=message, padx=20, pady=20, wraplength=400)
        msg_label.pack()

        # Button frame
        btn_frame = tk.Frame(dialog, padx=10, pady=10)
        btn_frame.pack()

        result = [None]  # Use list to allow modification in nested function

        def on_button(choice):
            result[0] = choice
            dialog.destroy()
            if root_created:
                root.destroy()

        # Create buttons (left to right)
        btn1 = tk.Button(btn_frame, text=button1, command=lambda: on_button(1), width=15)
        btn1.pack(side=tk.LEFT, padx=5)

        btn2 = tk.Button(btn_frame, text=button2, command=lambda: on_button(2), width=15)
        btn2.pack(side=tk.LEFT, padx=5)

        btn3 = tk.Button(btn_frame, text=button3, command=lambda: on_button(3), width=15)
        btn3.pack(side=tk.LEFT, padx=5)

        # Set default button
        if default_button == 1:
            btn1.focus_set()
        elif default_button == 2:
            btn2.focus_set()
        else:
            btn3.focus_set()

        # Center dialog on screen
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')

        dialog.transient(root)
        dialog.grab_set()
        root.wait_window(dialog)

        return result[0]

    except Exception as e:
        print(f"Error: Could not show dialog: {e}")
        return None


# Test functions
if __name__ == "__main__":
    print("Testing native dialogs...")

    print("\n1. Testing info dialog...")
    show_info("Test Info", "This is an info message")

    print("\n2. Testing warning dialog...")
    show_warning("Test Warning", "This is a warning message")

    print("\n3. Testing error dialog...")
    show_error("Test Error", "This is an error message")

    print("\n4. Testing yes/no dialog...")
    result = ask_yes_no("Test Question", "Do you want to continue?")
    print(f"   Result: {result}")

    print("\n5. Testing three-choice dialog...")
    result = ask_three_choice(
        "Test Three Choice",
        "What would you like to do?",
        "Option 1",
        "Option 2",
        "Option 3",
        default_button=2
    )
    print(f"   Result: {result}")

    print("\nAll tests complete!")
