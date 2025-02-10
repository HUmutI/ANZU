import os
import shutil
from tkinter import Tk, Label, Button, StringVar, Radiobutton, filedialog, Frame
from PIL import Image, ImageTk

# Global variables
current_image_index = 0
current_id = 15001
images = []
source_folder = ""
target_folder = ""

# Function to load images from the source folder
def load_images():
    global images
    images = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Function to move and rename the image
def process_image(new_name):
    global current_image_index, current_id

    # Get current image path
    current_image = images[current_image_index]
    current_image_path = os.path.join(source_folder, current_image)

    # Create new image path
    new_image_name = f"{new_name}_{current_id}.jpg"
    new_image_path = os.path.join(target_folder, new_image_name)

    # Move the image
    shutil.move(current_image_path, new_image_path)

    # Increment ID
    current_id += 1

# Function to display the current image
def display_image():
    global current_image_index

    if current_image_index < len(images):
        current_image_path = os.path.join(source_folder, images[current_image_index])
        img = Image.open(current_image_path)
        img.thumbnail((667, 667))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)

        label_image.configure(image=img_tk)
        label_image.image = img_tk

        # Update progress label
        label_progress.configure(text=f"Image {current_image_index + 1} of {len(images)}")
    else:
        label_image.configure(text="No images remaining.")
        label_progress.configure(text="All images processed.")

# Function to go to the next image
def next_image():
    global current_image_index

    if current_image_index >= len(images):
        label_image.configure(text="All images processed.")
        label_progress.configure(text="All images processed.")
        button_next.configure(state="disabled")
        return

    # Collect all selected values
    new_name = f"{var_aircraft_type.get()}_{var_background.get()}_{var_weather.get()}_{var_sunlight.get()}_{var_perspective.get()}_{var_color.get()}_{var_size.get()}"
    process_image(new_name)

    if current_image_index < len(images) - 1:
        current_image_index += 1
        display_image()
    else:
        label_image.configure(text="All images processed.")
        label_progress.configure(text="All images processed.")
        button_next.configure(state="disabled")

# Function to select folders
def select_folders():
    global source_folder, target_folder

    source_folder = filedialog.askdirectory(title="Select Source Folder")
    target_folder = filedialog.askdirectory(title="Select Target Folder")

    if source_folder and target_folder:
        load_images()
        if images:
            display_image()
        else:
            label_image.configure(text="No images found in the source folder.")
            label_progress.configure(text="")

# Initialize GUI
root = Tk()
root.title("Image Renamer")

# Variables for selections
var_aircraft_type = StringVar(value="rc")
var_background = StringVar(value="yesil")
var_weather = StringVar(value="acik")
var_sunlight = StringVar(value="ogle")
var_perspective = StringVar(value="yan")
var_color = StringVar(value="gri")
var_size = StringVar(value="orta")

# GUI Layout
button_frame = Frame(root)
button_frame.pack(pady=7)

button_select = Button(button_frame, text="Select Folders", command=select_folders, font=("Arial", 12), width=13)
button_select.grid(row=0, column=0, padx=7)

button_next = Button(button_frame, text="Next Image", command=next_image, font=("Arial", 12), width=13)
button_next.grid(row=0, column=1, padx=7)

label_progress = Label(root, text="", font=("Arial", 13))  # Label for progress
label_progress.pack(pady=7)

label_image = Label(root)
label_image.pack(pady=13)

Label(root, text="Select Image Properties:", font=("Arial", 13)).pack()

# Properties Layout
properties_frame = Frame(root)
properties_frame.pack(pady=7)

# Aircraft Type
aircraft_frame = Frame(properties_frame)
aircraft_frame.grid(row=0, column=0, padx=13)
Label(aircraft_frame, text="Aircraft Type:", font=("Arial", 12)).pack()
for text, value in [("RC", "rc"), ("UAV", "uav"), ("Uçak", "ucak"), ("Yolcu", "yolcu"), ("Delta", "delta")]:
    Radiobutton(aircraft_frame, text=text, variable=var_aircraft_type, value=value, font=("Arial", 10)).pack(anchor="w")

# Background
background_frame = Frame(properties_frame)
background_frame.grid(row=0, column=1, padx=13)
Label(background_frame, text="Background:", font=("Arial", 12)).pack()
for text, value in [("Yeşil", "yesil"), ("Bina", "bina"), ("Dağ", "dag"), ("Asfalt", "asfalt")]:
    Radiobutton(background_frame, text=text, variable=var_background, value=value, font=("Arial", 10)).pack(anchor="w")

# Weather
weather_frame = Frame(properties_frame)
weather_frame.grid(row=0, column=2, padx=13)
Label(weather_frame, text="Weather:", font=("Arial", 12)).pack()
for text, value in [("Az Bulutlu", "az-bulut"), ("Kapalı", "kapali"), ("Açık", "acik")]:
    Radiobutton(weather_frame, text=text, variable=var_weather, value=value, font=("Arial", 10)).pack(anchor="w")

# Sunlight
sunlight_frame = Frame(properties_frame)
sunlight_frame.grid(row=0, column=3, padx=13)
Label(sunlight_frame, text="Sunlight:", font=("Arial", 12)).pack()
for text, value in [("Öğle", "ogle"), ("Akşam", "aksam")]:
    Radiobutton(sunlight_frame, text=text, variable=var_sunlight, value=value, font=("Arial", 10)).pack(anchor="w")

# Perspective
perspective_frame = Frame(properties_frame)
perspective_frame.grid(row=0, column=4, padx=13)
Label(perspective_frame, text="Perspective:", font=("Arial", 12)).pack()
for text, value in [("Yan", "yan"), ("Arka", "arka"), ("Alt", "alt"), ("Üst", "ust")]:
    Radiobutton(perspective_frame, text=text, variable=var_perspective, value=value, font=("Arial", 10)).pack(anchor="w")

# Color
color_frame = Frame(properties_frame)
color_frame.grid(row=0, column=5, padx=13)
Label(color_frame, text="Color:", font=("Arial", 12)).pack()
for text, value in [("Siyah", "siyah"), ("Gri", "gri"), ("Beyaz", "beyaz"), ("Kırmızı", "kirmizi"), ("Yeşil", "yesil"), ("Mavi", "mavi")]:
    Radiobutton(color_frame, text=text, variable=var_color, value=value, font=("Arial", 10)).pack(anchor="w")

# Size
size_frame = Frame(properties_frame)
size_frame.grid(row=0, column=6, padx=13)
Label(size_frame, text="Size:", font=("Arial", 12)).pack()
for text, value in [("Büyük", "buyuk"), ("Orta", "orta"), ("Küçük", "kucuk")]:
    Radiobutton(size_frame, text=text, variable=var_size, value=value, font=("Arial", 10)).pack(anchor="w")

# Run the application
root.mainloop()
