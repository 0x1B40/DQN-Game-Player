import numpy as np
import cv2
import random
import logging
import win32gui
import win32con
import win32api
import pyautogui
from mss import mss
import ctypes
from ctypes import wintypes

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def capture_screen(window):
    try:
        with mss() as sct:
            return np.array(sct.grab(window))
    except Exception as e:
        logger.error(f"Screen capture failed: {e}")
        return None

def preprocess_frame(frame):
    try:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    except Exception as e:
        logger.error(f"Frame preprocessing failed: {e}")
        return frame

def focus_window(window_handle):
    try:
        if win32gui.IsIconic(window_handle):
            win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window_handle)
        win32gui.SetActiveWindow(window_handle)
        pyautogui.sleep(0.2)  # Ensure focus for emulators
        logger.debug(f"Window focused: {win32gui.GetWindowText(window_handle)}")
    except Exception as e:
        logger.error(f"Failed to focus window: {e}")

def send_input_key(vk_code, press=True):
    try:
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD),
                        ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_ulong)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD),
                        ("ki", KEYBDINPUT)]

        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002

        input_struct = INPUT()
        input_struct.type = INPUT_KEYBOARD
        input_struct.ki.wVk = vk_code
        input_struct.ki.wScan = 0
        input_struct.ki.dwFlags = 0 if press else KEYEVENTF_KEYUP
        input_struct.ki.time = 0
        input_struct.ki.dwExtraInfo = 0

        ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))
        logger.debug(f"Sent key (VK: {vk_code}) via SendInput {'press' if press else 'release'}")
    except Exception as e:
        logger.error(f"SendInput failed: {e}")

def execute_action(action, window_behaviors, window_positions, monitor, screen_window_handle=None):
    if not screen_window_handle:
        logger.error("No valid window handle provided")
        return

    logger.debug(f"Executing action: {action}")

    # Focus the emulator window
    focus_window(screen_window_handle)

    key_map = {
        "up": win32con.VK_UP,
        "down": win32con.VK_DOWN,
        "left": win32con.VK_LEFT,
        "right": win32con.VK_RIGHT,
        "space": win32con.VK_SPACE,
        "enter": win32con.VK_RETURN,
        "ctrlleft": win32con.VK_CONTROL,
        "ctrlright": win32con.VK_CONTROL,
        "shiftleft": win32con.VK_SHIFT,
        "shiftright": win32con.VK_SHIFT,
        "altleft": win32con.VK_MENU,
        "altright": win32con.VK_MENU,
        "z": 0x5A,  # Z key (SNES jump)
        "x": 0x58,  # X key (SNES run)
    }

    rect = win32gui.GetWindowRect(screen_window_handle)
    window_left, window_top = rect[0], rect[1]

    try:
        if action.startswith("left_click") or action.startswith("right_click"):
            if window_behaviors.get(action) == "Random" and monitor["width"] > 0:
                x = random.randint(monitor["left"], monitor["left"] + monitor["width"])
                y = random.randint(monitor["top"], monitor["top"] + monitor["height"])
            elif window_behaviors.get(action) == "Fixed" and action in window_positions:
                x, y = window_positions[action]
            else:
                logger.error(f"Invalid click action configuration: {action}")
                return

            # Adjust for window position
            screen_x, screen_y = x + window_left, y + window_top

            if action.startswith("left_click"):
                down_msg = win32con.WM_LBUTTONDOWN
                up_msg = win32con.WM_LBUTTONUP
                flags = win32con.MK_LBUTTON
                pyautogui_action = "left"
            else:
                down_msg = win32con.WM_RBUTTONDOWN
                up_msg = win32con.WM_RBUTTONUP
                flags = win32con.MK_RBUTTON
                pyautogui_action = "right"

            lparam = (screen_y << 16) | (screen_x & 0xFFFF)

            # Try PostMessage
            win32gui.PostMessage(screen_window_handle, down_msg, flags, lparam)
            win32gui.PostMessage(screen_window_handle, up_msg, 0, lparam)
            logger.debug(f"Sent mouse {pyautogui_action} click via PostMessage at ({screen_x}, {screen_y})")

            # Fallback to pyautogui
            try:
                pyautogui.click(x=screen_x, y=screen_y, button=pyautogui_action)
                logger.debug(f"Sent mouse {pyautogui_action} click via pyautogui at ({screen_x}, {screen_y})")
            except Exception as e:
                logger.error(f"pyautogui click failed: {e}")

        else:
            vk_code = key_map.get(action)
            if not vk_code and len(action) == 1:
                vk_code = win32api.VkKeyScan(action) & 0xFF

            if vk_code:
                # Try SendInput (DirectInput-compatible)
                send_input_key(vk_code, press=True)
                pyautogui.sleep(0.05)  # Brief hold
                send_input_key(vk_code, press=False)

                # Try PostMessage
                win32gui.PostMessage(screen_window_handle, win32con.WM_KEYDOWN, vk_code, 0)
                win32gui.PostMessage(screen_window_handle, win32con.WM_KEYUP, vk_code, 0)
                logger.debug(f"Sent key {action} (VK: {vk_code}) via PostMessage")

                # Fallback to pyautogui
                try:
                    pyautogui_key = action.lower() if action in key_map else action
                    pyautogui.press(pyautogui_key)
                    logger.debug(f"Sent key {pyautogui_key} via pyautogui")
                except Exception as e:
                    logger.error(f"pyautogui key press failed: {e}")
            else:
                logger.error(f"Invalid key action: {action}")

    except Exception as e:
        logger.error(f"Action execution failed: {e}")

def get_reward(prev_frame, curr_frame, reward_frame):
    try:
        reward_frame = cv2.cvtColor(reward_frame, cv2.COLOR_RGB2GRAY)
        reward_frame = cv2.resize(reward_frame, (84, 84)) / 255.0
        diff = np.mean(np.abs(prev_frame - reward_frame))
        reward = diff * 100  # Adjusted scaling
        if diff < 0.001:  # Detect static screen
            reward = -1.0
        logger.debug(f"Computed reward: {reward}")
        return reward
    except Exception as e:
        logger.error(f"Reward computation failed: {e}")
        return 0.0