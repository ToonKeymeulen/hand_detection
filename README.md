# Hand Sign Detection with MediaPipe

This project implements real-time hand sign detection using MediaPipe and OpenCV. It can recognize multiple hand signs including open hand, surfing sign, peace sign, and includes a bonus wave detection feature.

## Features

- Real-time hand sign detection using webcam
- Supports multiple signs:
  - Open hand
  - Surfing sign (🤙)
  - Peace sign (✌️)
  - Custom sign (bonus)
- Wave motion detection
- REST API endpoint for image-based prediction

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python app.py
```

This will start the web server and open your webcam for real-time detection.

### API Endpoint

Send POST requests to `/predict` with an image in JSON format:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image": "base64_encoded_image"}' http://localhost:5000/predict
```

## Project Structure

- `app.py`: Main application file with Flask server and endpoints
- `hand_detector.py`: Core hand detection and sign classification logic
- `utils.py`: Utility functions for image processing
- `benchmark/`: Directory containing benchmark code and test images
- `requirements.txt`: Project dependencies

## Benchmarking

The model has been benchmarked on a dataset of 10 images. Run the benchmark:

```bash
python benchmark/run_benchmark.py
``` 