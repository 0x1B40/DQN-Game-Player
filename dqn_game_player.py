import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import mss
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import time
from PIL import Image, ImageTk

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, input_shape, n_actions, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9, gamma=0.99, learning_rate=0.00025, buffer_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = DQN(input_shape, n_actions).to(self.device)
        self.target_model = DQN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# GUI Application with Multiple Fixed Mouse Clicks
class GameLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DQN Game Learning Agent")
        self.running = False
        self.agent = None
        self.actions = ["up", "down", "left", "right", "space"]  # Default actions
        self.mouse_behaviors = {}  # Tracks "Fixed" or "Random" for mouse actions
        self.mouse_positions = {}  # Tracks (x, y) for "Fixed" mouse actions
        self.mouse_click_counters = {"left_click": 0, "right_click": 0}  # Tracks number of fixed clicks
        self.monitor = {"top": 0, "left": 0, "width": 800, "height": 600}
        self.input_shape = (1, 84, 84)  # Grayscale, resized
        self.episode = 0
        self.total_reward = 0
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.last_key = None
        self.selecting_mouse_pos = False
        self.pending_mouse_action = None
        self.selection_mode = tk.StringVar(value="Select Window")
        self.sct = mss.mss()
        
        # GUI Elements
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control Buttons
        ttk.Button(self.frame, text="Start", command=self.start_learning).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Stop", command=self.stop_learning).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.frame, text="Refresh", command=self.refresh_preview).grid(row=0, column=2, padx=5, pady=5)
        
        # Selection Mode
        ttk.Label(self.frame, text="Selection Mode:").grid(row=1, column=0, sticky=tk.W)
        ttk.OptionMenu(self.frame, self.selection_mode, "Select Window", "Select Window", "Select Mouse Position").grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        # Input Customization
        ttk.Label(self.frame, text="Actions:").grid(row=2, column=0, sticky=tk.W)
        self.action_listbox = tk.Listbox(self.frame, height=5, width=40)
        self.action_listbox.grid(row=2, column=1, rowspan=3, columnspan=2, padx=5, pady=5)
        for action in self.actions:
            self.action_listbox.insert(tk.END, action)
        
        ttk.Label(self.frame, text="Last Key Pressed:").grid(row=5, column=0, sticky=tk.W)
        self.key_label = ttk.Label(self.frame, text="None")
        self.key_label.grid(row=5, column=1, columnspan=2, padx=5, pady=5)
        
        ttk.Label(self.frame, text="Mouse Behavior:").grid(row=6, column=0, sticky=tk.W)
        self.mouse_behavior = tk.StringVar(value="Fixed")
        ttk.OptionMenu(self.frame, self.mouse_behavior, "Fixed", "Fixed", "Random").grid(row=6, column=1, columnspan=2, padx=5, pady=5)
        
        ttk.Button(self.frame, text="Add Key", command=self.add_key_action).grid(row=7, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Remove Selected", command=self.remove_action).grid(row=7, column=1, columnspan=2, padx=5, pady=5)
        
        # Screen Selection Canvas
        self.canvas = tk.Canvas(self.frame, width=400, height=300, bg="white")
        self.canvas.grid(row=8, column=0, columnspan=3, pady=5)
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        # Monitor Coordinates Display
        self.coord_label = ttk.Label(self.frame, text="Selected Region: top=0, left=0, width=800, height=600")
        self.coord_label.grid(row=9, column=0, columnspan=3, pady=5)
        
        # Status Display
        self.status_label = ttk.Label(self.frame, text="Status: Stopped")
        self.status_label.grid(row=10, column=0, columnspan=3, pady=5)
        
        self.episode_label = ttk.Label(self.frame, text="Episode: 0")
        self.episode_label.grid(row=11, column=0, columnspan=3, pady=5)
        
        self.reward_label = ttk.Label(self.frame, text="Total Reward: 0")
        self.reward_label.grid(row=12, column=0, columnspan=3, pady=5)
        
        # Screen Preview
        self.preview_label = ttk.Label(self.frame)
        self.preview_label.grid(row=13, column=0, columnspan=3, pady=5)
        
        self.sct = mss.mss()
        pyautogui.PAUSE = 0.01
        
        # Bind key press detection
        self.root.bind("<KeyPress>", self.detect_key)
        
        # Initialize screen preview
        self.update_screen_preview()
    
    def update_screen_preview(self):
        # Capture entire screen
        monitor = self.sct.monitors[1]  # Primary monitor
        frame = np.array(self.sct.grab(monitor))
        img = Image.fromarray(frame)
        canvas_width, canvas_height = 400, 300
        img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.screen_photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screen_photo)
    
    def refresh_preview(self):
        # Clear any existing selection rectangle and refresh canvas
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.start_x = None
        self.start_y = None
        self.selecting_mouse_pos = False
        self.pending_mouse_action = None
        self.update_screen_preview()
    
    def detect_key(self, event):
        # Map tkinter key symbols to pyautogui keys
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
        if self.selection_mode.get() == "Select Window":
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
        else:  # Select Mouse Position
            self.handle_mouse_click(event, "left_click")
    
    def handle_right_click(self, event):
        if self.selection_mode.get() == "Select Mouse Position":
            self.handle_mouse_click(event, "right_click")
    
    def handle_mouse_click(self, event, action_base):
        if self.mouse_behavior.get() == "Fixed" and self.monitor["width"] > 0:
            # Start selecting position for fixed click
            self.selecting_mouse_pos = True
            self.pending_mouse_action = action_base
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red")
        else:
            # Add random mouse action (only one per type)
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
            # Screen area selection
            end_x, end_y = event.x, event.y
            monitor = self.sct.monitors[1]
            screen_width, screen_height = monitor["width"], monitor["height"]
            canvas_width, canvas_height = 400, 300
            scale_x = screen_width / canvas_width
            scale_y = screen_height / canvas_height
            
            top = int(min(self.start_y, end_y) * scale_y)
            left = int(min(self.start_x, end_x) * scale_x)
            width = int(abs(end_x - self.start_x) * scale_x)
            height = int(abs(end_y - self.start_y) * scale_y)
            
            self.monitor = {"top": top, "left": left, "width": width, "height": height}
            self.coord_label.configure(text=f"Selected Region: top={top}, left={left}, width={width}, height={height}")
            self.start_x = None
            self.start_y = None
            self.canvas.delete(self.rect)
            self.rect = None
        elif self.selecting_mouse_pos:
            # Store fixed mouse position
            action_base = self.pending_mouse_action
            if action_base:
                # Generate unique action identifier
                self.mouse_click_counters[action_base] += 1
                action = f"{action_base}_{self.mouse_click_counters[action_base]}"
                # Convert canvas coordinates to screen coordinates relative to game area
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
        if selection and len(self.actions) > 1:  # Ensure at least one action remains
            display_text = self.action_listbox.get(selection[0])
            action = display_text.split(" ")[0] if "(" in display_text else display_text
            self.actions.remove(action)
            self.action_listbox.delete(selection[0])
            if action in self.mouse_behaviors:
                del self.mouse_behaviors[action]
                if action in self.mouse_positions:
                    del self.mouse_positions[action]
    
    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def execute_action(self, action_idx):
        action = self.actions[action_idx]
        if action.startswith("left_click") or action.startswith("right_click"):
            if self.mouse_behaviors.get(action) == "Random" and self.monitor["width"] > 0:
                # Move to random position in game area
                x = random.randint(self.monitor["left"], self.monitor["left"] + self.monitor["width"])
                y = random.randint(self.monitor["top"], self.monitor["top"] + self.monitor["height"])
                pyautogui.moveTo(x, y)
            elif self.mouse_behaviors.get(action) == "Fixed" and action in self.mouse_positions:
                # Move to fixed position
                x, y = self.mouse_positions[action]
                pyautogui.moveTo(x, y)
            button = "left" if action.startswith("left_click") else "right"
            pyautogui.click(button=button)
        else:
            pyautogui.press(action)
    
    def get_reward(self, prev_frame, curr_frame):
        # Simple reward based on pixel differences
        diff = np.mean(np.abs(prev_frame - curr_frame))
        return diff * 0.1  # Scale for stability
    
    def update_preview(self, frame):
        img = Image.fromarray(frame)
        img = img.resize((200, 150), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
    
    def start_learning(self):
        if self.running or self.monitor["width"] == 0 or not self.actions:
            return
        self.running = True
        self.status_label.configure(text="Status: Running")
        
        # Initialize agent with current number of actions
        self.agent = DQNAgent(self.input_shape, len(self.actions))
        self.episode = 0
        self.total_reward = 0
        
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
                # Capture screen
                frame = np.array(self.sct.grab(self.monitor))
                state = self.preprocess_frame(frame)
                
                # Update preview
                self.update_preview(frame)
                
                # Get action
                action = self.agent.get_action(state)
                self.execute_action(action)
                
                # Capture next frame
                time.sleep(0.1)  # Simulate game frame rate
                next_frame = np.array(self.sct.grab(self.monitor))
                next_state = self.preprocess_frame(next_frame)
                
                # Calculate reward
                reward = self.get_reward(state, next_state) if prev_frame is not None else 0
                self.total_reward += reward
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                self.agent.train()
                
                # Update target network periodically
                if episode_steps % 100 == 0:
                    self.agent.update_target()
                
                # Update GUI
                self.episode_label.configure(text=f"Episode: {self.episode}")
                self.reward_label.configure(text=f"Total Reward: {self.total_reward:.2f}")
                
                prev_frame = state
                episode_steps += 1
                
                # Check for episode termination (simplified)
                if reward < 0.01:  # Assume no change means game over
                    done = True
                
                self.root.update()
            
            # Schedule next episode
            if self.running:
                self.root.after(100, learning_loop)
        
        self.root.after(100, learning_loop)
    
    def stop_learning(self):
        self.running = False
        self.status_label.configure(text="Status: Stopped")

if __name__ == "__main__":
    root = tk.Tk()
    app = GameLearningApp(root)
    root.mainloop()