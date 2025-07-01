import win32gui
import win32con
import win32api
import time

def simulate_alt_press():
    win32api.keybd_event(0x12, 0, 0, 0)
    time.sleep(0.05)
    win32api.keybd_event(0x12, 0, win32con.KEYEVENTF_KEYUP, 0)


def force_focus_window(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        simulate_alt_press()
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        win32gui.SetWindowPos(
            hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
        )
        win32gui.SetWindowPos(
            hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
        )


def focus_env_window(title_part="pygame"):
    def enumHandler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title_part.lower() in title.lower():

                force_focus_window(hwnd)
    win32gui.EnumWindows(enumHandler, None)