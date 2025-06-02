# main.py
import tkinter as tk
from modules.gui import GameScreenApp

if __name__ == "__main__":
    root = tk.Tk()
    app = GameScreenApp(root)
    root.mainloop()