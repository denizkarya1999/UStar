# UStar

UStar is an Android application that detects and tracks UOID (Underwater Optical Identification) tags using the device camera. UOID tags are optical markers used for underwater navigation. For example, a swimmer can pull out their phone underwater, open the UStar app, and the app will detect nearby UOID tags to show where to go and how far the destination is.

The app performs on-device machine learning to estimate both distance and orientation of the tag and displays the results directly in an ARCore overlay in real time.

## Features

### UOID Tag Tracking
- Detects and tracks UOID tags, which serve as underwater navigation markers.
- Enables swimmers and divers to understand their direction and distance to a target location by reading UOID tags placed underwater.
- Uses ML models to infer:
  - Relative distance to the tag
  - Viewing orientation of the tag (e.g., North, South, East, West)
- Designed for stable recognition of UOID tag geometry and colored reference points.

### Live Tracking
- Real-time UOID distance estimation  
- Real-time UOID orientation estimation  
- Continuous camera preview  
- Built-in AR overlay always active during tracking  
- Ability to:
  - Save the current frame  
  - Stop/resume tracking  
  - Switch between cameras  

### AR Mode
- Always-on ARCore overlay panel (“tablet”)
- Continuously displays:
  - Current distance  
  - Current orientation  
- Panel rotates dynamically based on device movement and camera pose

### Local Picture Inference
- Select any image from the device gallery
- Apply CycleGAN denoising to improve underwater visibility
- Run optical ranging and orientation models on the UOID tag within the picture
- View original and processed outputs side-by-side

### Additional Screens
- Settings page for configuration and local inference tools
- About page with version, release date, and developer information

## Machine Learning
- CycleGAN model for underwater image denoising and color correction
- ResNet-18 model for optical ranging (distance classification)
- ResNet-18 model for orientation classification (direction labeling)
- All models run fully on-device through PyTorch Mobile

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
