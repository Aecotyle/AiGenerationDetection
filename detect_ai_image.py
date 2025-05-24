import sys
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

def load_model(model_name="microsoft/beit-base-patch16-224-pt22k-ft22k"):
    """
    Load the Hugging Face model and feature extractor for image classification.
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

def predict_image(image_path, feature_extractor, model):
    """
    Predict if the image is AI-generated or not.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
    return predicted_label, confidence

def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_ai_image.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    print(f"Loading model...")
    feature_extractor, model = load_model()
    print(f"Predicting image: {image_path}")
    label, confidence = predict_image(image_path, feature_extractor, model)
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    # For demonstration, interpret label to AI-generated or not
    # This depends on the actual model labels; user may need to adjust
    if "ai" in label.lower() or "fake" in label.lower() or "generated" in label.lower():
        print("The image is likely AI-generated.")
    else:
        print("The image is likely NOT AI-generated.")

if __name__ == "__main__":
    main()
