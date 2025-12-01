# UStar

UStar is an Android application that performs real-time distance and orientation estimation using on-device machine learning and displays results through both a live camera interface and an ARCore overlay.

## Features

### Live Tracking
- Real-time distance classification
- Real-time orientation classification
- Live camera preview
- Save Frame, Stop Tracking, and Switch Camera actions

### AR Mode
- ARCore-based overlay panel
- Displays current distance and orientation
- Optional billboard effect for facing the camera

### Local Picture Inference
- Select images from the device gallery
- Run CycleGAN denoising
- Run optical ranging and orientation models
- View original and processed images

### Additional Screens
- Settings page for accessing local inference
- About page with version and developer information

## Machine Learning
- CycleGAN for denoising
- ResNet-18 for optical ranging
- ResNet-18 for orientation guidance
- Models executed through PyTorch Mobile

## Tech Stack
- Kotlin
- ARCore
- OpenGL ES 2.0
- PyTorch Android
- Camera2 / SharedCamera API

## Installation
1. Clone the repository:
   git clone https://github.com/<your-username>/UStar.git
2. Open the project in Android Studio.
3. Place the required .pt model files into the assets folder.
4. Build and run on a physical ARCore-capable Android device.