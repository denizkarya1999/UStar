# UStar

UStar is an Android application that tracks UOID (Underwater Optical Identification) tags using the device camera, runs on-device machine learning models for distance and orientation estimation, and displays the results in both a live camera view and an ARCore overlay.

## Features

### UOID Tag Tracking
- Detects and tracks UOID tag structures in the camera feed
- Uses ML models to infer the relative distance to the tag
- Uses ML models to classify the viewing orientation of the tag
- Designed for consistent recognition of UOID tag geometry and colored positioning elements

### Live Tracking
- Real-time UOID distance estimation
- Real-time UOID orientation estimation
- Live camera preview
- Options to save the current frame, stop tracking, and switch cameras

### AR Mode
- ARCore-based overlay panel ("tablet")
- Shows the current distance and orientation inferred from the UOID tag
- Panel can optionally face the camera (billboard mode)

### Local Picture Inference
- Select an image from the device gallery
- Run CycleGAN denoising on the image
- Run optical ranging and orientation models on the UOID tag within the image
- View original and processed outputs

### Additional Screens
- Settings page for accessing local inference
- About page with version, release date, and developer information

## Machine Learning
- CycleGAN model for denoising UOID images
- ResNet-18 model for optical ranging (distance classification)
- ResNet-18 model for orientation guidance (direction classification)
- Models executed on-device via PyTorch Mobile

## Tech Stack
- Kotlin
- Camera2 / SharedCamera API
- ARCore
- OpenGL ES 2.0
- PyTorch Android
- Material UI Components

## Installation
1. Clone the repository:
   git clone https://github.com/<your-username>/UStar.git
2. Open the project in Android Studio.
3. Place required .pt model files in the assets directory.
4. Build and run on an ARCore-supported physical device.