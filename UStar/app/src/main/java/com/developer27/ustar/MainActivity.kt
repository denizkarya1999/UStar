package com.developer27.ustar

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import org.opencv.android.OpenCVLoader
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.graphics.drawable.BitmapDrawable
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.preference.PreferenceManager
import android.provider.MediaStore
import android.util.Log
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.ustar.camera.CameraHelper
import com.developer27.ustar.databinding.ActivityMainBinding
import com.developer27.ustar.machinelearning.Denoising_CycleGAN
import com.developer27.ustar.machinelearning.DynaSpa.MiniDynaSpaPreprocessor
import com.developer27.ustar.machinelearning.Optical_Ranging_ResNet18
import com.developer27.ustar.machinelearning.Orientation_Guidance_ResNet18
import com.developer27.ustar.videoprocessing.VideoProcessor
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * MainActivity
 * ------------
 * Handles:
 *  - Runtime permissions (camera & mic)
 *  - Camera2 preview and frame display
 *  - Frame processing via [VideoProcessor]
 *
 * UI Summary:
 *  - Start/Stop Tracking toggles processing and recording
 *  - Switch Camera flips between front/back lenses
 *  - Settings/About buttons open separate activities
 */
class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null

    // State flags
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    // --- Global variable for prediction ---
    companion object {
        var currentPrediction: String = ""
    }

    // Camera permissions
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) {
            if (allPermissionsGranted()) cameraHelper.openCamera()
            else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, w: Int, h: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) processFrameWithVideoProcessor()
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Lifecycle                                                         */
    /* ------------------------------------------------------------------ */
    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Keep screen awake and portrait locked
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Basic services
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        cameraHelper = CameraHelper(this, viewBinding)
        videoProcessor = VideoProcessor(this)
        viewBinding.processedFrameView.visibility = View.GONE

        // Load OpenCV libraries
        if (OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV loaded successfully")
        } else {
            Log.e("OpenCV", "OpenCV load failed")
        }

        // Load the CycleGAN based denoising model on startup
        Denoising_CycleGAN.load(this)

        // Load DynaSpa Making Processor Model
        MiniDynaSpaPreprocessor.load(this)

        // Load the ResNet-18 based optical ranging model on startup
        Optical_Ranging_ResNet18.loadModel(this)

        // Load the ResNet-18 based orientation model on startup
        Orientation_Guidance_ResNet18.loadModel(this)

        // Tap title to open website
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.zhangxiao.me/")))
        }

        // Request permissions if not granted
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
                val cam = perms[Manifest.permission.CAMERA] ?: false
                val mic = perms[Manifest.permission.RECORD_AUDIO] ?: false
                if (cam && mic) {
                    if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
                    else viewBinding.viewFinder.surfaceTextureListener = textureListener
                } else {
                    Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
                }
            }

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
            else viewBinding.viewFinder.surfaceTextureListener = textureListener
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        /* ---------------------- UI BUTTONS ---------------------- */
        // Start/Stop Processing
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording() else startProcessingAndRecording()
        }

        // Collect Dataset: save current frame (processed if available, else raw preview)
        viewBinding.collectDatasetButton.setOnClickListener {
            val saved = saveCurrentFrame()
            if (saved) {
                Toast.makeText(this, "Frame saved to Pictures/UStar/Frames", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Could not save frame.", Toast.LENGTH_SHORT).show()
            }
        }

        // ResNet-18 Inference: The prediction will update the label
        val predictionText: TextView = findViewById(R.id.predictionText)
        predictionText.text = currentPrediction

        // Switch camera
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }

        // About / Settings
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutUStarActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Setup zoom gestures
        cameraHelper.setupZoomControls()
    }

    /* ------------------------------------------------------------------ */
    /*  Resume Mechanisms                                                 */
    /* ------------------------------------------------------------------ */
    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) {
            cameraHelper.openCamera()
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Pause Mechanisms                                                  */
    /* ------------------------------------------------------------------ */
    override fun onPause() {
        isProcessing = false
        if (isRecording) stopProcessingAndRecording()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    /* ------------------------------------------------------------------ */
    /*  Camera Controls                                                   */
    /* ------------------------------------------------------------------ */
    private var isFrontCamera = false

    private fun switchCamera() {
        if (isRecording) stopProcessingAndRecording()
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    /* ------------------------------------------------------------------ */
    /*  Start Processing                                                  */
    /* ------------------------------------------------------------------ */
    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true

        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE


        // Make Collect Dataset visible while processing
        viewBinding.collectDatasetButton.visibility = View.VISIBLE

        videoProcessor?.reset()
    }

    /* ------------------------------------------------------------------ */
    /*  Stop Processing                                                   */
    /* ------------------------------------------------------------------ */
    private fun stopProcessingAndRecording(){
        isRecording = false
        isProcessing = false

        viewBinding.processedFrameView.setImageBitmap(null)
        viewBinding.processedFrameView.visibility = View.GONE
        val predictionText: TextView = findViewById(R.id.predictionText)
        predictionText.text = ""

        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // Hide Collect Dataset when processing stops
        viewBinding.collectDatasetButton.visibility = View.GONE

        // Trigger the AR viewer
        try {
            val intent = Intent(
                this,
                com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java
            )
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(
                this,
                "Unable to launch AR Activity: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Frame Processing                                                  */
    /* ------------------------------------------------------------------ */
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        videoProcessor?.processFrame(bitmap) { result ->
            runOnUiThread {
                if (!isProcessing) {
                    isProcessingFrame = false
                    return@runOnUiThread
                }

                result?.let { processed ->
                    viewBinding.processedFrameView.setImageBitmap(processed)
                    val predictionText: TextView = findViewById(R.id.predictionText)
                    predictionText.text = currentPrediction
                }

                isProcessingFrame = false
            }
        }
    }

    //  Save Current Frame when processing
    private fun saveCurrentFrame(): Boolean {
        // Prefer the processed bitmap shown in processedFrameView
        val processedDrawable = viewBinding.processedFrameView.drawable
        val bitmap: Bitmap? = when (processedDrawable) {
            is BitmapDrawable -> processedDrawable.bitmap
            else -> viewBinding.viewFinder.bitmap   // Fallback to live preview
        }

        if (bitmap == null) {
            Log.w("MainActivity", "saveCurrentFrame: bitmap is null")
            return false
        }

        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(System.currentTimeMillis())
            val fileName = "UStar_${timestamp}.png"

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/UStar/Frames")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }

            val resolver = contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: return false

            resolver.openOutputStream(uri)?.use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }

            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)

            true
        } catch (e: Exception) {
            Log.e("MainActivity", "saveCurrentFrame failed: ${e.message}", e)
            false
        }
    }
}
