import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import pyautogui
import win32gui
from .dqn_agent import DQNAgent
from .utils import capture_screen, preprocess_frame, execute_action, get_reward

class GameLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN Game Learning Agent")
        self.running = False
        self.recording = False
        self.agent = None
        self.actions = ["space"]  # Default for Flappy Bird
        self.mouse_behaviors = {}
        self.mouse_positions = {}
        self.mouse_click_counters = {"left_click": 0, "right_click": 0}
        self.monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
        self.game_window_handle = None
        self.input_shape = (1, 84, 84)
        self.episode = 0
        self.total_reward = 0
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.last_key = None
        self.selecting_mouse_pos = False
        self.pending_mouse_action = None
        self.selection_mode = tk.StringVar(value="Select Window")
        self.demo_states = []
        self.demo_actions = []
        
        # Set pyautogui pause (for key detection)
        pyautogui.PAUSE = 0.01
        
        # GUI Elements
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Buttons
        ttk.Button(self.frame, text="Start", command=self.start_learning).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Stop", command=self.stop_learning).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.frame, text="Refresh", command=self.refresh_preview).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.frame, text="Record Demo", command=self.start_recording).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(self.frame, text="Reset Model", command=self.reset_model).grid(row=0, column=4, padx=5, pady=5)
        
        # Selection Mode
        ttk.Label(self.frame, text="Selection Mode:").grid(row=1, column=0, sticky=tk.W)
        ttk.OptionMenu(self.frame, self.selection_mode, "Select Window", "Select Window", "Select Mouse Position").grid(row=1, column=1, columnspan=4, padx=5, pady=5)
        
        # Input Customization
        ttk.Label(self.frame, text="Actions:").grid(row=2, column=0, sticky=tk.W)
        self.action_listbox = tk.Listbox(self.frame, height=5, width=40)
        self.action_listbox.grid(row=2, column=1, rowspan=3, columnspan=4, padx=5, pady=5)
        for action in self.actions:
            self.action_listbox.insert(tk.END, action)
        
        ttk.Label(self.frame, text="Last Key Pressed:").grid(row=5, column=0, sticky=tk.W)
        self.key_label = ttk.Label(self.frame, text="None")
        self.key_label.grid(row=5, column=1, columnspan=4, padx=5, pady=5)
        
        ttk.Label(self.frame, text="Mouse Behavior:").grid(row=6, column=0, sticky=tk.W)
        self.mouse_behavior = tk.StringVar(value="Fixed")
        ttk.OptionMenu(self.frame, self.mouse_behavior, "Fixed", "Fixed", "Random").grid(row=6, column=1, columnspan=4, padx=5, pady=5)
        
        ttk.Button(self.frame, text="Add Key", command=self.add_key_action).grid(row=7, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Remove Selected", command=self.remove_action).grid(row=7, column=1, columnspan=4, padx=5, pady=5)
        
        # Screen Selection Canvas
        self.canvas = tk.Canvas(self.frame, width=400, height=300, bg="white")
        self.canvas.grid(row=8, column=0, columnspan=5, pady=5)
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        # Monitor Coordinates Display
        self.coord_label = ttk.Label(self.frame, text="Selected Region: top=0, left=0, width=800, height=600")
        self.coord_label.grid(row=9, column=0, columnspan=5, pady=5)
        
        # Status Display
        self.status_label = ttk.Label(self.frame, text="Status: Stopped")
        self.status_label.grid(row=10, column=0, columnspan=5, pady=5)
        
        self.episode_label = ttk.Label(self.frame, text="Episode: 0")
        self.episode_label.grid(row=11, column=0, columnspan=5, pady=5)
        
        self.reward_label = ttk.Label(self.frame, text="Total Reward: 0")
        self.reward_label.grid(row=12, column=0, columnspan=5, pady=5)
        
        # Screen Preview
        self.preview_label = ttk.Label(self.frame)
        self.preview_label.grid(row=13, column=0, columnspan=5, pady=5)
        
        # Bind key press detection
        self.root.bind("<KeyPress>", self.detect_key)
        
        # Initialize screen preview
        self.update_screen_preview()
    
    def update_screen_preview(self):
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        frame = capture_screen(monitor)
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
        self.selecting_mouse_pos = False
        self.pending_mouse_action = None
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
        
        # Record action during demo
        if self.recording and key in self.actions:
            frame = capture_screen(self.monitor)
            state = preprocess_frame(frame)
            action_idx = self.actions.index(key)
            self.demo_states.append(state)
            self.demo_actions.append(action_idx)
            execute_action(key, self.mouse_behaviors, self.mouse_positions, self.monitor, self.game_window_handle)
    
    def add_key_action(self):
        if self.last_key and self.last_key not in self.actions:
            self.actions.append(self.last_key)
            self.action_listbox.insert(tk.END, self.last_key)
    
    def handle_left_click(self, event):
        if self.selection_mode.get() == "Select Window":
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
        else:
            self.handle_mouse_click(event, "left_click")
    
    def handle_right_click(self, event):
        if self.selection_mode.get() == "Select Mouse Position":
            self.handle_mouse_click(event, "right_click")
    
    def handle_mouse_click(self, event, action_base):
        if self.mouse_behavior.get() == "Fixed" and self.monitor["width"] > 0:
            self.selecting_mouse_pos = True
            self.pending_mouse_action = action_base
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
        else:
            action = f"{action_base}_random" if action_base in ["left_click", "right_click"] else action_base
            if action not in self.actions:
                self.actions.append(action)
                self.mouse_behaviors[action] = "Random"
                self.action_listbox.insert(tk.END, f"{action_base} (Random)")
    
    def update_selection(self, event):
        if self.selection_mode.get() == "Select Window" and self.start_x is not None:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        elif self.selecting_mouse_pos:
            self.canvas.coords(self.rect, event.x-3, event.y-3, event.x+3, event.y+3)
    
    def end_selection(self, event):
        if self.selection_mode.get() == "Select Window" and self.start_x is not None:
            end_x, end_y = event.x, event.y
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            screen_width, screen_height = monitor["width"], monitor["height"]
            canvas_width, canvas_height = 400, 300
            scale_x = screen_width / canvas_width
            scale_y = screen_height / canvas_height
            
            top = int(min(self.start_y, end_y) * scale_y)
            left = int(min(self.start_x, end_x) * scale_x)
            width = int(abs(end_x - self.start_x) * scale_x)
            height = int(abs(end_y - self.start_y) * scale_y)
            
            self.monitor = {"top": top, "left": left, "width": width, "height": height}
            self.game_window_handle = win32gui.WindowFromPoint((left + width // 2, top + height // 2))
            self.coord_label.configure(text=f"Selected Region: top={top}, left={left}, width={width}, height={height}")
            self.start_x = None
            self.start_y = None
            self.canvas.delete(self.rect)
            self.rect = None
        elif self.selecting_mouse_pos:
            action_base = self.pending_mouse_action
            if action_base:
                self.mouse_click_counters[action_base] += 1
                action = f"{action_base}_{self.mouse_click_counters[action_base]}"
                canvas_width, canvas_height = 400, 300
                scale_x = self.monitor["width"] / canvas_width if self.monitor["width"] > 0 else 1
                scale_y = self.monitor["height"] / canvas_height if self.monitor["height"] > 0 else 1
                x = int(event.x * scale_x) + self.monitor["left"]
                y = int(event.y * scale_y) + self.monitor["top"]
                self.actions.append(action)
                self.mouse_behaviors[action] = "Fixed"
                self.mouse_positions[action] = (x, y)
                self.action_listbox.insert(tk.END, f"{action_base}_{self.mouse_click_counters[action_base]} (Fixed: {x}, {y})")
            self.selecting_mouse_pos = False
            self.pending_mouse_action = None
            self.canvas.delete(self.rect)
            self.rect = None
    
    def remove_action(self):
        selection = self.action_listbox.curselection()
        if selection and len(self.actions) > 1:
            display_text = self.action_listbox.get(selection[0])
            action = display_text.split(" ")[0] if "(" in display_text else display_text
            self.actions.remove(action)
            self.action_listbox.delete(selection[0])
            if action in self.mouse_behaviors:
                del self.mouse_behaviors[action]
                if action in self.mouse_positions:
                    del self.mouse_positions[action]
    
    def update_preview(self, frame):
        img = Image.fromarray(frame)
        img = img.resize((200, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
    
    def start_recording(self):
        if self.running or self.recording or self.monitor["width"] == 0 or not self.game_window_handle:
            return
        self.recording = True
        self.status_label.configure(text="Status: Recording Demo")
        self.demo_states = []
        self.demo_actions = []
    
    def stop_learning(self):
        self.running = False
        self.recording = False
        self.status_label.configure(text="Status: Stopped")
    
    def reset_model(self):
        self.running = False
        self.recording = False
        self.agent = None
        self.demo_states = []
        self.demo_actions = []
        self.episode = 0
        self.total_reward = 0
        self.episode_label.configure(text="Episode: 0")
        self.reward_label.configure(text="Total Reward: 0")
        self.status_label.configure(text="Status: Stopped")
    
    def start_learning(self):
        if (self.running or self.recording or 
            self.monitor["width"] == 0 or 
            not self.actions or 
            not self.game_window_handle):
            return
        self.running = True
        self.status_label.configure(text="Status: Running")
        
        # Initialize or reinitialize agent if needed
        if self.agent is None:
            self.agent = DQNAgent(self.input_shape, len(self.actions))
            if self.demo_states and self.demo_actions:
                self.agent.pretrain_on_demos(self.demo_states, self.demo_actions)
        
        def learning_loop():
            if not self.running:
                return
            self.episode += 1
            self.total_reward = 0
            prev_frame = None
            done = False
            episode_steps = 0
            max_steps = 1000
            
            while not done and episode_steps < max_steps:
                if not self.running:
                    break
                frame = capture_screen(self.monitor)
                state = preprocess_frame(frame)
                
                self.update_preview(frame)
                
                action_idx = self.agent.get_action(state)
                action = self.actions[action_idx]
                execute_action(action, self.mouse_behaviors, self.mouse_positions, self.monitor, self.game_window_handle)
                
                time.sleep(0.1)
                next_frame = capture_screen(self.monitor)
                next_state = preprocess_frame(next_frame)
                
                reward = get_reward(state, next_state, next_frame) if prev_frame is not None else 0
                self.total_reward += reward
                
                self.agent.store_transition(state, action_idx, reward, next_state, done)
                
                self.agent.train()
                
                if episode_steps % 100 == 0:
                    self.agent.update_target()
                
                self.episode_label.configure(text=f"Episode: {self.episode}")
                self.reward_label.configure(text=f"Total Reward: {self.total_reward:.2f}")
                
                prev_frame = state
                episode_steps += 1
                
                if reward < 0:  # Game over detected
                    done = True
                
                self.root.update()
            
            if self.running:
                self.root.after(100, learning_loop)
        
        self.root.after(100, learning_loop)