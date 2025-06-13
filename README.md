# Plant Disease Detection App

This repository contains a simple web application for detecting plant diseases from leaf images. The application is built with [Streamlit](https://streamlit.io/) and uses a ResNet-18 model trained on common plant disease datasets.

## Features

- Easy-to-use interface for uploading leaf images
- Pretrained model included for instant predictions
- Supports detection of 15 common plant diseases in peppers, potatoes, and tomatoes

## Installation

1. Ensure you have Python 3.8 or newer installed.
2. Install the required packages:
   ```bash
   pip install streamlit torch torchvision pillow
   ```

## Running the App

Start the Streamlit server with:

```bash
streamlit run app.py
```

This will open the web interface in your browser where you can upload a plant leaf photo and get a prediction.

## Files

- `app.py` – Streamlit application code
- `class_names.json` – List of class labels used by the model
- `plant_disease_resnet18.pth` – Pretrained model weights
- `LICENSE` – MIT License information

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

