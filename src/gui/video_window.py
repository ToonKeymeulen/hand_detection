import cv2
import tkinter as tk
from PIL import Image, ImageTk
from queue import Queue
from typing import Optional

class VideoWindow:
    def __init__(self, frame_queue: Queue, should_exit: Optional[callable] = None):
        self.frame_queue = frame_queue
        self.should_exit = should_exit
        
        self.root = tk.Tk()
        self.root.title("Hand Sign Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        
        self.sign_label = tk.Label(self.root, text="No signs detected", font=("Arial", 14))
        self.sign_label.pack()
        
        self.update_frame()
    
    def on_closing(self):
        if self.should_exit:
            self.should_exit.set()
        self.root.quit()
        self.root.destroy()
    
    def update_frame(self):
        try:
            if not self.frame_queue.empty():
                frame, signs = self.frame_queue.get_nowait()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=image)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                self.sign_label.config(text=f"Detected signs: {', '.join(signs)}" if signs else "No signs detected")
        except Exception:
            pass
        
        if not self.should_exit or not self.should_exit.is_set():
            self.root.after(30, self.update_frame)
    
    def run(self):
        self.root.mainloop() 