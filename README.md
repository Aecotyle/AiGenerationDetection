# AI Generation Detection Tool

This tool uses Hugging Face models to detect if an image is AI-generated or not.

## Setup

1. Create a Python virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On Linux/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the detection script with the path to the image you want to check:

```bash
python detect_ai_image.py path/to/image.jpg
```

The script will output whether the image is likely AI-generated or not based on the model prediction.

## Notes

- The default model used is `microsoft/beit-base-patch16-224-pt22k-ft22k`. You may replace it with a more suitable model for AI-generated image detection if available.
- The prediction interpretation is basic and depends on the model's label names.
- For better accuracy, consider fine-tuning a model specifically for AI-generated image detection.
