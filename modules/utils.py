import numpy as np
import cv2
import mss
import random
import win32gui
import win32con
import win32api

def capture_screen(monitor):
    with mss.mss() as sct:
        return np.array(sct.grab(monitor))

def preprocess_frame(frame):
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def execute_action(action, mouse_behaviors, mouse_positions, monitor, game_window_handle=None):
    if not game_window_handle:
        return
    
    key_map = {
        "space": win32con.VK_SPACE,
        "enter": win32con.VK_RETURN,
        "ctrlleft": win32con.VK_CONTROL,
        "ctrlright": win32con.VK_CONTROL,
        "shiftleft": win32con.VK_SHIFT,
        "shiftright": win32con.VK_SHIFT,
        "altleft": win32con.VK_MENU,
        "altright": win32con.VK_MENU,
    }
    
    rect = win32gui.GetWindowRect(game_window_handle)
    window_left, window_top = rect[0], rect[1]
    
    if action.startswith("left_click") or action.startswith("right_click"):
        if mouse_behaviors.get(action) == "Random" and monitor["width"] > 0:
            x = random.randint(monitor["left"], monitor["left"] + monitor["width"])
            y = random.randint(monitor["top"], monitor["top"] + monitor["height"])
        elif mouse_behaviors.get(action) == "Fixed" and action in mouse_positions:
            x, y = mouse_positions[action]
        else:
            return
        
        client_point = (x - window_left, y - window_top)
        
        if action.startswith("left_click"):
            down_msg = win32con.WM_LBUTTONDOWN
            up_msg = win32con.WM_LBUTTONUP
            flags = win32con.MK_LBUTTON
        else:
            down_msg = win32con.WM_RBUTTONDOWN
            up_msg = win32con.WM_RBUTTONUP
            flags = win32con.MK_RBUTTON
        
        lparam = (client_point[1] << 16) | (client_point[0] & 0xFFFF)
        
        win32gui.PostMessage(game_window_handle, down_msg, flags, lparam)
        win32gui.PostMessage(game_window_handle, up_msg, 0, lparam)
    else:
        vk_code = key_map.get(action)
        if not vk_code:
            vk_code = win32api.VkKeyScan(action) & 0xFF if len(action) == 1 else None
        if vk_code:
            win32gui.PostMessage(game_window_handle, win32con.WM_KEYDOWN, vk_code, 0)
            win32gui.PostMessage(game_window_handle, win32con.WM_KEYUP, vk_code, 0)

def get_reward(prev_frame, curr_frame, raw_frame):
    # Survival reward
    reward = 1.0
    
    # Check for game over (high brightness indicates game-over screen)
    brightness = np.mean(raw_frame)
    if brightness > 200:  # Adjust threshold based on Flappy Bird's game-over screen
        reward = -100.0
    
    return reward