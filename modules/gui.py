import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import pyautogui
import win32gui
import logging
from .dqn_agent import DQNAgent
from .utils import capture_screen, preprocess_frame, execute_action, get_reward

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameScreenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rainbow DQN Game Learning Agent")
        self.running = False
        self.agent = None
        self.actions = ["up", "down", "left", "right", "space"]
        self.window_behaviors = {}
        self.window_positions = {}
        self.window_click_counters = {"left_click": 0, "right_click": 0}
        self.window = {"top": 0, "left": 0, "width": 800, "height": 600}
        self.screen_window_handle = None
        self.input_shape = (1, 84, 84)
        self.episode = 0
        self.total_reward = 0.0
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.last_key = None
        self.selecting_window_pos = False
        self.pending_window_action = None
        self.selection_mode = tk.StringVar(value="Select Window")

        pyautogui.PAUSE = 0.01

        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(self.frame, text="Start", command=self.start_learning).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Stop", command=self.stop_learning).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.frame, text="Refresh", command=self.refresh_preview).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.frame, text="Reset Model", command=self.reset_model).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(self.frame, text="Test Action", command=self.test_action).grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(self.frame, text="Selection Mode:").grid(row=1, column=0, sticky=tk.W)
        ttk.OptionMenu(self.frame, self.selection_mode, "Select Window", "Select Window", "Select Mouse Position").grid(row=1, column=1, columnspan=4, padx=5, pady=5)

        ttk.Label(self.frame, text="Actions:").grid(row=2, column=0, sticky=tk.W)
        self.action_listbox = tk.Listbox(self.frame, height=5, width=40)
        self.action_listbox.grid(row=2, column=1, rowspan=3, columnspan=4, padx=5, pady=5)
        for action in self.actions:
            self.action_listbox.insert(tk.END, action)

        ttk.Label(self.frame, text="Last Key Pressed:").grid(row=5, column=0, sticky=tk.W)
        self.key_label = ttk.Label(self.frame, text="None")
        self.key_label.grid(row=5, column=1, columnspan=4, padx=5, pady=5)

        ttk.Label(self.frame, text="Mouse Behavior:").grid(row=6, column=0, sticky=tk.W)
        self.window_behavior = tk.StringVar(value="Fixed")
        ttk.OptionMenu(self.frame, self.window_behavior, "Fixed", "Fixed", "Random").grid(row=6, column=1, columnspan=4, padx=5, pady=5)

        ttk.Button(self.frame, text="Add Key", command=self.add_key_action).grid(row=7, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Remove Selected", command=self.remove_action).grid(row=7, column=1, columnspan=4, padx=5, pady=5)

        self.canvas = tk.Canvas(self.frame, width=400, height=300, bg="white")
        self.canvas.grid(row=8, column=0, columnspan=5, pady=5)
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

        self.game_coord_label = ttk.Label(self.frame, text="Game Region: top=0, left=0, width=800, height=600")
        self.game_coord_label.grid(row=9, column=0, columnspan=5, pady=5)

        self.status_label = ttk.Label(self.frame, text="Status: Stopped")
        self.status_label.grid(row=10, column=0, columnspan=5, pady=5)

        self.episode_label = ttk.Label(self.frame, text="Episode: 0")
        self.episode_label.grid(row=11, column=0, columnspan=5, pady=5)

        self.reward_label = ttk.Label(self.frame, text="Total Reward: 0.0")
        self.reward_label.grid(row=12, column=0, columnspan=5, pady=5)

        self.preview_label = ttk.Label(self.frame)
        self.preview_label.grid(row=13, column=0, columnspan=5, pady=5)

        self.root.bind("<KeyPress>", self.detect_key)
        self.update_screen_preview()

    def update_screen_preview(self):
        window = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        frame = capture_screen(window)
        if frame is not None:
            img = Image.fromarray(frame)
            canvas_width, canvas_height = 400, 300
            img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            self.screen_photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screen_photo)

    def refresh_preview(self):
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.start_x = None
        self.start_y = None
        self.selecting_window_pos = False
        self.pending_window_action = None
        self.update_screen_preview()

    def detect_key(self, event):
        key_map = {
            "Return": "enter",
            "Control_L": "ctrlleft",
            "Control_R": "ctrlright",
            "Shift_L": "shiftleft",
            "Shift_R": "shiftright",
            "Alt_L": "altleft",
            "Alt_R": "altright",
            " ": "space"
        }
        key = event.keysym
        if key in key_map:
            key = key_map[key]
        elif len(key) == 1 and key.lower() in pyautogui.KEYBOARD_KEYS:
            key = key.lower()
        elif key in pyautogui.KEYBOARD_KEYS:
            pass
        else:
            return
        self.last_key = key
        self.key_label.configure(text=f"Last Key Pressed: {key}")

    def add_key_action(self):
        if self.last_key and self.last_key not in self.actions:
            self.actions.append(self.last_key)
            self.action_listbox.insert(tk.END, self.last_key)

    def handle_left_click(self, event):
        mode = self.selection_mode.get()
        if mode == "Select Window":
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
        else:
            self.handle_window_click(event, "left_click")

    def handle_right_click(self, event):
        if self.selection_mode.get() == "Select Mouse Position":
            self.handle_window_click(event, "right_click")

    def handle_window_click(self, event, action_base):
        if self.window_behavior.get() == "Fixed" and self.window["width"] > 0:
            self.selecting_window_pos = True
            self.pending_window_action = action_base
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
        else:
            action = f"{action_base}_random" if action_base in ["left_click", "right_click"] else action_base
            if action not in self.actions:
                self.actions.append(action)
                self.window_behaviors[action] = "Random"
                self.action_listbox.insert(tk.END, f"{action_base} (Random)")

    def update_selection(self, event):
        mode = self.selection_mode.get()
        if mode == "Select Window" and self.start_x is not None:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        elif self.selecting_window_pos:
            self.canvas.coords(self.rect, event.x-3, event.y-3, event.x+3, event.y+3)

    def end_selection(self, event):
        mode = self.selection_mode.get()
        if mode == "Select Window" and self.start_x is not None:
            end_x, end_y = event.x, event.y
            window = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            screen_width, screen_height = window["width"], window["height"]
            canvas_width, canvas_height = 400, 300
            scale_x = screen_width / canvas_width
            scale_y = screen_height / canvas_height

            top = int(min(self.start_y, end_y) * scale_y)
            left = int(min(self.start_x, end_x) * scale_x)
            width = int(abs(end_x - self.start_x) * scale_x)
            height = int(abs(end_y - self.start_y) * scale_y)

            self.window = {"top": top, "left": left, "width": width, "height": height}
            try:
                self.screen_window_handle = win32gui.WindowFromPoint((left + width // 2, top + height // 2))
                window_title = win32gui.GetWindowText(self.screen_window_handle)
                logger.debug(f"Selected window handle: {self.screen_window_handle}, Title: {window_title}")
                if not window_title:
                    logger.warning("Selected window has no title, may not be valid")
                    self.screen_window_handle = None
            except Exception as e:
                logger.error(f"Window selection failed: {e}")
                self.screen_window_handle = None
            self.game_coord_label.configure(text=f"Game Region: top={top}, left={left}, width={width}, height={height}")

            self.start_x = None
            self.start_y = None
            self.canvas.delete(self.rect)
            self.rect = None
        elif self.selecting_window_pos:
            action_base = self.pending_window_action
            if action_base:
                self.window_click_counters[action_base] += 1
                action = f"{action_base}_{self.window_click_counters[action_base]}"
                canvas_width, canvas_height = 400, 300
                scale_x = self.window["width"] / canvas_width if self.window["width"] > 0 else 1
                scale_y = self.window["height"] / canvas_height if self.window["height"] > 0 else 1
                x = int(event.x * scale_x) + self.window["left"]
                y = int(event.y * scale_y) + self.window["top"]
                self.actions.append(action)
                self.window_behaviors[action] = "Fixed"
                self.window_positions[action] = (x, y)
                self.action_listbox.insert(tk.END, f"{action_base}_{self.window_click_counters[action_base]} (Fixed: {x}, {y})")
            self.selecting_window_pos = False
            self.pending_window_action = None
            self.canvas.delete(self.rect)
            self.rect = None

    def remove_action(self):
        selection = self.action_listbox.curselection()
        if selection and len(self.actions) > 1:
            display_text = self.action_listbox.get(selection[0])
            action = display_text.split(" ")[0] if "(" in display_text else display_text
            self.actions.remove(action)
            self.action_listbox.delete(selection[0])
            if action in self.window_behaviors:
                del self.window_behaviors[action]
                if action in self.window_positions:
                    del self.window_positions[action]

    def update_preview(self, frame):
        if frame is not None:
            img = Image.fromarray(frame)
            img = img.resize((200, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo

    def test_action(self):
        if not self.screen_window_handle or not self.actions:
            logger.error("No window handle or actions configured for testing")
            self.status_label.configure(text="Status: No window or actions")
            return
        action = self.actions[0]
        execute_action(action, self.window_behaviors, self.window_positions, self.window, self.screen_window_handle)
        self.status_label.configure(text=f"Status: Tested action {action}")

    def stop_learning(self):
        self.running = False
        self.status_label.configure(text="Status: Stopped")

    def reset_model(self):
        self.running = False
        self.agent = None
        self.episode = 0
        self.total_reward = 0.0
        self.episode_label.configure(text="Episode: 0")
        self.reward_label.configure(text="Total Reward: 0.0")
        self.status_label.configure(text="Status: Stopped")

    def start_learning(self):
        if (self.running or
            self.window["width"] <= 0 or
            not self.actions or
            not self.screen_window_handle):
            logger.error("Cannot start learning: Invalid window, no actions, or already running")
            self.status_label.configure(text="Status: Invalid configuration")
            return

        self.running = True
        self.status_label.configure(text="Status: Running")

        if self.agent is None:
            self.agent = DQNAgent(self.input_shape, len(self.actions))

        def learning_loop():
            if not self.running:
                return
            self.episode += 1
            self.total_reward = 0.0
            prev_frame = None
            done = False
            episode_steps = 0
            max_steps = 1000

            while not done and episode_steps < max_steps:
                if not self.running:
                    break
                frame = capture_screen(self.window)
                if frame is None:
                    logger.error("Failed to capture frame, stopping episode")
                    break
                state = preprocess_frame(frame)
                if state is None:
                    logger.error("Failed to preprocess frame, stopping episode")
                    break

                self.update_preview(frame)

                action_idx = self.agent.get_action(state)
                action = self.actions[action_idx]
                execute_action(action, self.window_behaviors, self.window_positions, self.window, self.screen_window_handle)

                time.sleep(0.1)
                next_frame = capture_screen(self.window)
                if next_frame is None:
                    logger.error("Failed to capture next frame, stopping episode")
                    break
                next_state = preprocess_frame(next_frame)
                if next_state is None:
                    logger.error("Failed to preprocess next frame, stopping episode")
                    break

                reward = get_reward(state, next_state, next_frame) if prev_frame is not None else 0.0
                self.total_reward += reward

                self.agent.store_transition(state, action_idx, reward, next_state, done)

                self.agent.train()

                if episode_steps % 100 == 0:
                    self.agent.update_target()

                self.episode_label.configure(text=f"Episode: {self.episode}")
                self.reward_label.configure(text=f"Total Reward: {self.total_reward:.2f}")

                prev_frame = state
                episode_steps += 1

                if reward < 0.1:
                    done = True

                self.root.update()

            if self.running:
                self.root.after(100, learning_loop)

        self.root.after(100, learning_loop)