# Traffic Image Segmentation Backend

This backend server provides an API for segmenting traffic images using a deep learning model (U-Net architecture).

## Features

- Image segmentation API endpoint
- Supports 12 different classes for traffic scene elements (road, car, pedestrian, etc.)
- Returns colored segmentation masks based on the trained model

## Requirements

- Python 3.8 or higher
- PyTorch 1.13.1
- Flask 2.0.1
- Other dependencies listed in requirements.txt

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure the model file `best_model.pth` is in the correct location (should be in the Back-End folder).

## Running the Server

To run the backend server, execute:

```bash
python app.py
```

The server will run on http://localhost:5000 by default.

## API Endpoints

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: `{"status": "ok"}`

### Image Segmentation

- **URL**: `/api/segment`
- **Method**: `POST`
- **Body**: Form data with an 'image' file
- **Response**: JSON object with:
  - `success`: Boolean indicating success/failure
  - `segmented_image`: Base64 encoded PNG image (with "data:image/png;base64," prefix)

## Color Map

The segmentation mask uses the following color mapping:

- Sky: (128, 128, 128)
- Building: (128, 0, 0)
- Pole: (192, 192, 128)
- Road: (128, 64, 128)
- Pavement: (60, 40, 222)
- Tree: (128, 128, 0)
- SignSymbol: (192, 128, 128)
- Fence: (64, 64, 128)
- Car: (64, 0, 128)
- Pedestrian: (64, 64, 0)
- Bicyclist: (0, 128, 192)
- Unlabeled: (0, 0, 0)

## Troubleshooting

If you encounter issues with the model loading, make sure:
1. The path to `best_model.pth` is correct
2. You have the correct PyTorch version installed
3. The model was trained with a compatible PyTorch version 