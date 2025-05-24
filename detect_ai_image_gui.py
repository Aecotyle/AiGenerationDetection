import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Reuse the model loading and prediction functions from detect_ai_image.py

def load_model(model_name="microsoft/beit-base-patch16-224-pt22k-ft22k"):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

def predict_image(image_path, feature_extractor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
    return predicted_label, confidence

class AIImageDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.feature_extractor, self.model = load_model()

        self.label = tk.Label(root, text="Select an image to detect if it is AI-generated or not.")
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_path:
            self.display_image(file_path)
            self.predict_and_display(file_path)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def predict_and_display(self, image_path):
        label, confidence = predict_image(image_path, self.feature_extractor, self.model)
        result_text = f"Prediction: {label} (Confidence: {confidence:.2f})\n"
        if "ai" in label.lower() or "fake" in label.lower() or "generated" in label.lower():
            result_text += "The image is likely AI-generated."
        else:
            result_text += "The image is likely NOT AI-generated."
        self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AIImageDetectorApp(root)
    root.mainloop()
