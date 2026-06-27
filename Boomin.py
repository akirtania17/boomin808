import customtkinter
import tkinter
from tkinter import filedialog
from pathlib import Path
import pedalboard
from pedalboard.io import AudioFile
import threading
import io
import requests
from PIL import Image

# --- SAKURA THEME COLORS ---
SAKURA_BG = "#FFF0F5"         # LavenderBlush - Main window background
SAKURA_FRAME = "#FFDDEE"      # Light pink for frames
SAKURA_BUTTON = "#F4ABC4"      # A soft, medium pink for buttons
SAKURA_HOVER = "#EAA0B9"      # A slightly darker pink for hover
SAKURA_SLIDER = "#E56B9F"     # A more vibrant pink for the slider progress
SAKURA_TEXT = "#3D2B32"        # A dark, moody pink-ish grey for text
SAKURA_ACCENT = "#D15C8D"      # Accent for progress bar

# --- Set up the main application window ---
customtkinter.set_appearance_mode("Light") # A light theme works better for pinks

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Configuration ---
        self.title("🌸 Sakura 808 Processor 🌸")
        self.geometry("550x700")
        self.configure(bg_color=SAKURA_BG)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Internal State Variables ---
        self.input_path = None
        self.output_path = None
        self.audio_data = None
        self.sample_rate = 0
        self.processed_audio = None

        # --- GUI WIDGETS (Themed) ---

        # Frame for File I/O
        self.file_frame = customtkinter.CTkFrame(self, fg_color=SAKURA_FRAME, corner_radius=15)
        self.file_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.file_frame.grid_columnconfigure(1, weight=1)

        self.load_button = customtkinter.CTkButton(
            self.file_frame, text="Load 808 (.wav)", command=self.load_file,
            fg_color=SAKURA_BUTTON, text_color=SAKURA_TEXT, hover_color=SAKURA_HOVER
        )
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        self.load_label = customtkinter.CTkLabel(self.file_frame, text="No file loaded.", anchor="w", text_color=SAKURA_TEXT)
        self.load_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.save_button = customtkinter.CTkButton(
            self.file_frame, text="Save Processed 808", command=self.save_file, state="disabled",
            fg_color=SAKURA_BUTTON, text_color=SAKURA_TEXT, hover_color=SAKURA_HOVER
        )
        self.save_button.grid(row=1, column=0, padx=10, pady=10)
        self.save_label = customtkinter.CTkLabel(self.file_frame, text="Process a file to save.", anchor="w", text_color=SAKURA_TEXT)
        self.save_label.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Frame for Effect Controls
        self.controls_frame = customtkinter.CTkFrame(self, fg_color=SAKURA_FRAME, corner_radius=15)
        self.controls_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        # --- THEMED SLIDERS ---
        slider_kwargs = {
            "button_color": SAKURA_BUTTON,
            "button_hover_color": SAKURA_HOVER,
            "progress_color": SAKURA_SLIDER
        }
        
        self.drive_label = customtkinter.CTkLabel(self.controls_frame, text="Distortion Drive (dB)", text_color=SAKURA_TEXT)
        self.drive_label.grid(row=0, column=0, padx=20, pady=(15,0), sticky="w")
        self.drive_slider = customtkinter.CTkSlider(self.controls_frame, from_=0, to=40, number_of_steps=40, command=self.update_processing, **slider_kwargs)
        self.drive_slider.set(22)
        self.drive_slider.grid(row=1, column=0, padx=20, pady=(0,20), sticky="ew")

        self.threshold_label = customtkinter.CTkLabel(self.controls_frame, text="Compressor Threshold (dB)", text_color=SAKURA_TEXT)
        self.threshold_label.grid(row=2, column=0, padx=20, pady=(10,0), sticky="w")
        self.threshold_slider = customtkinter.CTkSlider(self.controls_frame, from_=-40, to=0, number_of_steps=40, command=self.update_processing, **slider_kwargs)
        self.threshold_slider.set(-18)
        self.threshold_slider.grid(row=3, column=0, padx=20, pady=(0,20), sticky="ew")

        self.ratio_label = customtkinter.CTkLabel(self.controls_frame, text="Compressor Ratio", text_color=SAKURA_TEXT)
        self.ratio_label.grid(row=4, column=0, padx=20, pady=(10,0), sticky="w")
        self.ratio_slider = customtkinter.CTkSlider(self.controls_frame, from_=1, to=10, number_of_steps=18, command=self.update_processing, **slider_kwargs)
        self.ratio_slider.set(5)
        self.ratio_slider.grid(row=5, column=0, padx=20, pady=(0,20), sticky="ew")

        self.sub_boost_label = customtkinter.CTkLabel(self.controls_frame, text="Sub Boost (dB @ 80Hz)", text_color=SAKURA_TEXT)
        self.sub_boost_label.grid(row=6, column=0, padx=20, pady=(10,0), sticky="w")
        self.sub_boost_slider = customtkinter.CTkSlider(self.controls_frame, from_=0, to=12, number_of_steps=12, command=self.update_processing, **slider_kwargs)
        self.sub_boost_slider.set(6)
        self.sub_boost_slider.grid(row=7, column=0, padx=20, pady=(0,20), sticky="ew")

        self.punch_boost_label = customtkinter.CTkLabel(self.controls_frame, text="Punch Boost (dB @ 200Hz)", text_color=SAKURA_TEXT)
        self.punch_boost_label.grid(row=8, column=0, padx=20, pady=(10,0), sticky="w")
        self.punch_boost_slider = customtkinter.CTkSlider(self.controls_frame, from_=0, to=12, number_of_steps=12, command=self.update_processing, **slider_kwargs)
        self.punch_boost_slider.set(4)
        self.punch_boost_slider.grid(row=9, column=0, padx=20, pady=(0,20), sticky="ew")

        # --- Status Bar ---
        self.status_frame = customtkinter.CTkFrame(self, height=50, fg_color=SAKURA_FRAME, corner_radius=10)
        self.status_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        self.status_label = customtkinter.CTkLabel(self.status_frame, text="Load a file to begin.", text_color=SAKURA_TEXT)
        self.status_label.pack(side="left", padx=10)
        self.progress_bar = customtkinter.CTkProgressBar(self.status_frame, progress_color=SAKURA_ACCENT)
        self.progress_bar.pack(side="right", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget() # Hide it initially

    def load_file(self):
        self.input_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.input_path:
            try:
                with AudioFile(self.input_path) as f:
                    self.audio_data = f.read(f.frames)
                    self.sample_rate = f.samplerate
                self.load_label.configure(text=Path(self.input_path).name)
                self.status_label.configure(text="File loaded. Processing...")
                self.update_processing(None)
            except Exception as e:
                self.status_label.configure(text=f"Error: Could not read file. {e}")
    
    def update_processing(self, _):
        if self.audio_data is not None:
            self.save_button.configure(state="disabled")
            self.status_label.configure(text="Processing with new settings...")
            self.progress_bar.pack(side="right", padx=10, fill="x", expand=True)
            self.progress_bar.start()

            processing_thread = threading.Thread(target=self.process_audio_thread)
            processing_thread.daemon = True
            processing_thread.start()

    def process_audio_thread(self):
        # Get current values from all sliders
        drive = self.drive_slider.get()
        threshold = self.threshold_slider.get()
        ratio = self.ratio_slider.get()
        sub_boost = self.sub_boost_slider.get()
        # ** Bug Fix: Was self.punch_slider, now correctly self.punch_boost_slider **
        punch_boost = self.punch_boost_slider.get()

        board = pedalboard.Pedalboard([
            pedalboard.Distortion(drive_db=drive),
            pedalboard.Compressor(threshold_db=threshold, ratio=ratio, attack_ms=1.5, release_ms=120),
            pedalboard.EQ(low_shelf_frequency_hz=80, low_shelf_gain_db=sub_boost, low_shelf_q=0.7,
                          peak_frequency_hz=200, peak_gain_db=punch_boost, peak_q=1.2),
            pedalboard.Limiter(threshold_db=-1.0, release_ms=50)
        ])

        self.processed_audio = board.process(self.audio_data, self.sample_rate)
        self.after(100, self.on_processing_complete)

    def on_processing_complete(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="Ready to save! �")
        self.save_button.configure(state="normal")
        self.save_label.configure(text="Click 'Save' to choose a location.")

    def save_file(self):
        if self.processed_audio is not None:
            self.output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
            if self.output_path:
                try:
                    with AudioFile(self.output_path, 'w', self.sample_rate, self.processed_audio.shape[0]) as f:
                        f.write(self.processed_audio)
                    self.save_label.configure(text=f"Saved to {Path(self.output_path).name}")
                    self.status_label.configure(text="File saved successfully!")
                except Exception as e:
                    self.status_label.configure(text=f"Error: Could not save file. {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
