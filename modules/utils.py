import numpy as np
import cv2
import random
import logging
import win32gui
import win32con
import pyautogui
from mss import mss
from pynput.keyboard import Key, Controller
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log')
logger = logging.getLogger(__name__)

# Initialize pynput keyboard controller
keyboard = Controller()

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
        if not window_handle or not win32gui.IsWindow(window_handle):
            logger.error("Invalid or non-existent window handle")
            return False
        if win32gui.IsIconic(window_handle):
            win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
        for _ in range(3):  # Retry up to 3 times
            win32gui.SetForegroundWindow(window_handle)
            win32gui.SetActiveWindow(window_handle)
            time.sleep(0.2)
            current_foreground = win32gui.GetForegroundWindow()
            if current_foreground == window_handle:
                logger.debug(f"Window focused: {win32gui.GetWindowText(window_handle)}")
                return True
            logger.warning("Retrying window focus")
        logger.error(f"Failed to set focus: Current foreground is {win32gui.GetWindowText(current_foreground)}")
        # Fallback to Alt+Tab
        pyautogui.hotkey("alt", "tab")
        time.sleep(0.5)
        if win32gui.GetForegroundWindow() == window_handle:
            logger.debug(f"Window focused via Alt+Tab: {win32gui.GetWindowText(window_handle)}")
            return True
        logger.error("Failed to focus window after Alt+Tab")
        return False
    except Exception as e:
        logger.error(f"Failed to focus window: {e}")
        return False

def execute_action(action, window_behaviors, window_positions, monitor, screen_window_handle=None):
    if not screen_window_handle:
        logger.error("No valid window handle provided")
        return

    logger.debug(f"Executing action: {action}")

    # Ensure the window is focused
    if not focus_window(screen_window_handle):
        logger.error("Cannot execute action: Window focus failed")
        return

    # Map actions to pynput keys
    key_map = {
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "space": Key.space,
        "enter": Key.enter,
        "ctrlleft": Key.ctrl_l,
        "ctrlright": Key.ctrl_r,
        "shiftleft": Key.shift_l,
        "shiftright": Key.shift_r,
        "altleft": Key.alt_l,
        "altright": Key.alt_r,
        "z": "z",
        "x": "x",
    }

    try:
        if action.startswith("left_click") or action.startswith("right_click"):
            rect = win32gui.GetWindowRect(screen_window_handle)
            window_left, window_top = rect[0], rect[1]
            if window_behaviors.get(action) == "Random" and monitor["width"] > 0:
                x = random.randint(monitor["left"], monitor["left"] + monitor["width"])
                y = random.randint(monitor["top"], monitor["top"] + monitor["height"])
            elif window_behaviors.get(action) == "Fixed" and action in window_positions:
                x, y = window_positions[action]
            else:
                logger.error(f"Invalid click action configuration: {action}")
                return

            screen_x, screen_y = x + window_left, y + window_top
            pyautogui.click(x=screen_x, y=screen_y, button="left" if action.startswith("left_click") else "right")
            logger.debug(f"Sent mouse {'left' if action.startswith('left_click') else 'right'} click at ({screen_x}, {screen_y})")
        else:
            # Handle keyboard input with pynput
            key = key_map.get(action, action)  # Use action directly if not in key_map
            if key:
                try:
                    keyboard.press(key)
                    time.sleep(0.1)
                    keyboard.release(key)
                    logger.debug(f"Sent key {action} via pynput")
                except ValueError:
                    logger.error(f"Invalid key for pynput: {action}")
            else:
                logger.error(f"Invalid key action: {action}")
    except Exception as e:
        logger.error(f"Action execution failed: {e}")

def get_reward(prev_frame, curr_frame, reward_frame):
    try:
        reward_frame = cv2.cvtColor(reward_frame, cv2.COLOR_RGB2GRAY)
        reward_frame = cv2.resize(reward_frame, (84, 84)) / 255.0
        diff = np.mean(np.abs(prev_frame - reward_frame))
        reward = diff * 100
        if diff < 0.005:
            reward = -0.5
        logger.debug(f"Computed reward: {reward}")
        return reward
    except Exception as e:
        logger.error(f"Reward computation failed: {e}")
        return 0.0