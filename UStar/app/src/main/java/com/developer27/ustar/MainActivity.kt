package com.developer27.ustar

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import org.opencv.android.OpenCVLoader
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.graphics.drawable.BitmapDrawable
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
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
import androidx.preference.PreferenceManager
import com.developer27.ustar.camera.CameraHelper
import com.developer27.ustar.databinding.ActivityMainBinding
import com.developer27.ustar.videoprocessing.Settings
import com.developer27.ustar.videoprocessing.VideoProcessor
import java.text.SimpleDateFormat
import java.io.File
import java.util.Locale

/**
 * MainActivity
 * ------------
 * Handles:
 *  - Runtime camera permission
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
    private lateinit var cameraHelper: CameraHelper
    private var videoProcessor: VideoProcessor? = null

    // State flags
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false
    private var isCameraLifecycleActive = false

    // --- Global variable for prediction ---
    companion object {
        @Volatile
        var currentPrediction: String = ""
    }

    // Camera permissions
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA
    )

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var requestStoragePermissionLauncher: ActivityResultLauncher<String>

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) {
            openCameraIfReady()
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, w: Int, h: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            cameraHelper.closeCamera()
            return true
        }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) processFrameWithVideoProcessor()
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Lifecycle                                                         */
    /* ------------------------------------------------------------------ */
    override fun onCreate(savedInstanceState: Bundle?) {
        // Keep screen awake and portrait locked
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Basic services
        cameraHelper = CameraHelper(this, viewBinding)
        videoProcessor = VideoProcessor(this)
        viewBinding.processedFrameView.visibility = View.GONE

        Settings.selectedDenoiser = PreferenceManager.getDefaultSharedPreferences(this)
            .getString("pref_denoiser_type", "cyclegan")
            ?: "cyclegan"

        // Load OpenCV libraries
        if (OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV loaded successfully")
        } else {
            Log.e("OpenCV", "OpenCV load failed")
        }

        // Tap title to open website
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.zhangxiao.me/")))
        }

        // Request permissions if not granted
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
                val cam = perms[Manifest.permission.CAMERA] ?: false
                if (cam) {
                    openCameraIfReady()
                } else {
                    Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_SHORT).show()
                }
            }

        requestStoragePermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
                if (granted) saveCurrentFrameAndNotify()
                else Toast.makeText(this, "Storage permission is required to save frames.", Toast.LENGTH_SHORT).show()
            }

        viewBinding.viewFinder.surfaceTextureListener = textureListener

        /* ---------------------- UI BUTTONS ---------------------- */
        // Start/Stop Processing
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording(launchAr = true)
            else startProcessingAndRecording()
        }

        // Collect Dataset: save current frame (processed if available, else raw preview)
        viewBinding.collectDatasetButton.setOnClickListener {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                requestStoragePermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            } else {
                saveCurrentFrameAndNotify()
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
        isCameraLifecycleActive = true
        cameraHelper.startBackgroundThread()
        if (allPermissionsGranted()) {
            openCameraIfReady()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Pause Mechanisms                                                  */
    /* ------------------------------------------------------------------ */
    override fun onPause() {
        isCameraLifecycleActive = false
        isProcessing = false
        if (isRecording) stopProcessingAndRecording(launchAr = false)
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        videoProcessor?.close()
        videoProcessor = null
        super.onDestroy()
    }

    /* ------------------------------------------------------------------ */
    /*  Camera Controls                                                   */
    /* ------------------------------------------------------------------ */
    private var isFrontCamera = false

    private fun switchCamera() {
        if (!allPermissionsGranted()) {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            return
        }
        if (isRecording) stopProcessingAndRecording(launchAr = false)
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        openCameraIfReady()
    }

    @SuppressLint("MissingPermission")
    private fun openCameraIfReady() {
        if (!isCameraLifecycleActive || !allPermissionsGranted() || !viewBinding.viewFinder.isAvailable) {
            return
        }

        try {
            cameraHelper.openCamera()
        } catch (e: SecurityException) {
            Log.e("MainActivity", "Camera permission was revoked while opening the camera", e)
        }
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
    private fun stopProcessingAndRecording(launchAr: Boolean) {
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

        if (!launchAr) return

        // Trigger the AR viewer only when the user explicitly stops tracking.
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
    private fun saveCurrentFrameAndNotify() {
        val saved = saveCurrentFrame()
        val message = if (saved) {
            "Frame saved to Pictures/UStar/Frames"
        } else {
            "Could not save frame."
        }
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    @Suppress("DEPRECATION")
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

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/UStar/Frames")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }

                val resolver = contentResolver
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: return false
                val written = resolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                } == true
                if (!written) {
                    resolver.delete(uri, null, null)
                    return false
                }

                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            } else {
                val pictures = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                val destination = File(pictures, "UStar/Frames/$fileName")
                destination.parentFile?.mkdirs()
                destination.outputStream().use { out ->
                    if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) return false
                }
                MediaScannerConnection.scanFile(
                    this,
                    arrayOf(destination.absolutePath),
                    arrayOf("image/png"),
                    null
                )
            }

            true
        } catch (e: Exception) {
            Log.e("MainActivity", "saveCurrentFrame failed: ${e.message}", e)
            false
        }
    }
}
