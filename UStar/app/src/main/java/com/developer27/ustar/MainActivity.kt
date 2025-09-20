package com.developer27.ustar

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.preference.PreferenceManager
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.lifecycleScope
import com.developer27.ustar.camera.CameraHelper
import com.developer27.ustar.databinding.ActivityMainBinding
import com.developer27.ustar.videoprocessing.CycleGAN
import com.developer27.ustar.videoprocessing.ProcessedVideoRecorder
import com.developer27.ustar.videoprocessing.Settings
import com.developer27.ustar.videoprocessing.VideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

/**
 * MainActivity
 * ------------
 * Drives the end‑to‑end demo:
 *  - Handles runtime permissions (camera/mic).
 *  - Opens the Camera via [CameraHelper] and renders its preview in a TextureView.
 *  - For each preview frame, optionally runs the YOLO pipeline via [VideoProcessor].
 *  - Displays processed frames and optionally records them via [ProcessedVideoRecorder].
 *  - Loads the TFLite model on a background thread with NNAPI/GPU/CPU fallback.
 *
 * UI summary:
 *  - Start/Stop Tracking toggles processing + recording.
 *  - Switch Camera flips between back/front.
 *  - Settings/About launch separate screens.
 *  - Take Photo captures a still using [CameraHelper] (visibility driven by [Settings.ExportData.takePhoto]).
 */
class MainActivity : AppCompatActivity() {

    // ViewBinding for activity_main.xml
    private lateinit var viewBinding: ActivityMainBinding

    // App-scoped dependencies
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper

    // ML components
    private var yoloInterpreter: Interpreter? = null
    private var videoProcessor: VideoProcessor? = null

    // CycleGAN toggle state
    private var isCycleGanApplied = false
    private var cycleGanOriginalBitmap: Bitmap? = null
    private var prevProcessingStateForCycle = false
    private var cycleGanLoopJob: Job? = null

    // Optional processed-video recorder (disabled by default)
    private var processedVideoRecorder: ProcessedVideoRecorder? = null

    // Simple state flags that gate UI and frame processing
    private var isRecording = false             // "tracking/recording" toggle
    private var isProcessing = false            // whether we should process incoming frames
    private var isProcessingFrame = false       // guards concurrent frame processing calls
    private var shouldUpdateTakePhotoBtn = true // re-check visibility on resume once

    // Runtime permissions we require
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    // Modern permission launcher (Activity Result API)
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        /**
         * Rotation to degrees mapping for camera sensor vs. device UI rotation.
         * Not directly used in this file (kept as a handy reference/utility).
         */
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0,   90)
            append(Surface.ROTATION_90,  0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    /**
     * TextureView callbacks for the preview surface used by Camera2.
     * - When the surface becomes available, we open the camera (if we have permission).
     * - On every update (new frame), if processing is enabled, we ask [VideoProcessor] to process it.
     */
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, w: Int, h: Int) { /* no-op */ }

        // Return false to indicate that the app will handle surface release (TextureView keeps it)
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false

        // Called for every preview frame rendered into the TextureView
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) processFrameWithVideoProcessor()
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Lifecycle                                                         */
    /* ------------------------------------------------------------------ */

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Keep the screen on during preview/processing
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // Lock the demo to portrait (adjust if you need landscape support)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        // Android 12+ splash
        installSplashScreen()

        super.onCreate(savedInstanceState)

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Basic app services
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        // Helper classes that encapsulate camera & processing details
        cameraHelper   = CameraHelper(this, viewBinding)
        videoProcessor = VideoProcessor(this)

        // Hide the processed preview until tracking starts
        viewBinding.processedFrameView.visibility = View.GONE

        /*------ App / web link (title tap opens a website) ------*/
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.zhangxiao.me/")))
        }

        /*------ Permission launcher: ask once, then open camera if granted ------*/
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

        // Initial permission/camera bootstrapping
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
            else viewBinding.viewFinder.surfaceTextureListener = textureListener
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        /*------ UI buttons ------*/

        // Start/Stop (tracking + optional recording)
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording() else startProcessingAndRecording()
        }

        // Flip between back and front cameras
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }

        // About & Settings screens
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutUStarActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Still photo capture using CameraHelper; shows the captured bitmap in processedFrameView
        viewBinding.takePhotoButton.setOnClickListener {
            cameraHelper.takePhoto { file ->
                Toast.makeText(this, "Photo saved: ${file.absolutePath}", Toast.LENGTH_SHORT).show()
                val bmp = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
                viewBinding.processedFrameView.setImageBitmap(bmp)
                viewBinding.processedFrameView.visibility = View.GONE
                viewBinding.processedFrameView.setImageBitmap(null)
            }
        }

        // Show/hide Take Photo button based on Settings
        updateTakePhotoVisibility()

        /*------ Load the CycleGAN TorchLite model on a background thread ------*/
        CycleGAN.load(this)

        // Run once on a test asset image
        val (inFile, outFile) = CycleGAN.runOnAssetToPictures(
            this,
            "test.jpg",
            "test_out.png",
            "test_in.png"
        )

        Toast.makeText(
            this,
            "CycleGAN input and output samples are saved in:\nPictures/UStar/",
            Toast.LENGTH_LONG
        ).show()

        /*------ Load the (YOLO) TFLite model on a background thread ------*/
        loadTFLiteModelThreaded("YOLOv3_float32.tflite")

        // Trigger CycleGAN on button click
        viewBinding.cycleGanButton.setOnClickListener {
            if (!viewBinding.viewFinder.isAvailable) {
                Toast.makeText(this, "Camera not ready yet.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            if (!isCycleGanApplied) {
                // remember & pause the live pipeline so our overlay isn't overwritten
                prevProcessingStateForCycle = isProcessing
                isProcessing = false

                viewBinding.cycleGanButton.isEnabled = false

                lifecycleScope.launch {
                    try {
                        // Grab one frame immediately
                        val src: Bitmap? = withContext(Dispatchers.Main) {
                            viewBinding.viewFinder.getBitmap(256, 256)
                        }
                        if (src == null) {
                            Toast.makeText(this@MainActivity, "No frame to process.", Toast.LENGTH_SHORT).show()
                            isProcessing = prevProcessingStateForCycle
                            return@launch
                        }

                        cycleGanOriginalBitmap = src.copy(Bitmap.Config.ARGB_8888, false)

                        val stylized = withContext(Dispatchers.Default) { CycleGAN.run(src) }
                        withContext(Dispatchers.Main) {
                            viewBinding.processedFrameView.setImageBitmap(stylized)
                            viewBinding.processedFrameView.visibility = View.VISIBLE
                            isCycleGanApplied = true

                            // update button UI
                            viewBinding.cycleGanButton.text = "Stop CycleGAN"
                            viewBinding.cycleGanButton.backgroundTintList =
                                ContextCompat.getColorStateList(this@MainActivity, R.color.red)

                            Log.i("CycleGAN", "✅ Applied CycleGAN and displayed first result.")
                        }

                        // start periodic refresh while CycleGAN is applied
                        cycleGanLoopJob?.cancel()
                        cycleGanLoopJob = lifecycleScope.launch {
                            while (isActive && isCycleGanApplied) {
                                try {
                                    val srcLoop: Bitmap? = withContext(Dispatchers.Main) {
                                        viewBinding.viewFinder.getBitmap(256, 256)
                                    }
                                    if (srcLoop != null) {
                                        val stylizedLoop = withContext(Dispatchers.Default) { CycleGAN.run(srcLoop) }
                                        withContext(Dispatchers.Main) {
                                            if (isCycleGanApplied) {
                                                viewBinding.processedFrameView.setImageBitmap(stylizedLoop)
                                            }
                                        }
                                    }
                                } catch (t: Throwable) {
                                    Log.e("CycleGAN", "Periodic inference error: ${t.message}", t)
                                }
                                delay(1) // refresh every 0.001 seconds
                            }
                        }
                    } catch (t: Throwable) {
                        Log.e("CycleGAN", "CycleGAN failed: ${t.message}", t)
                        Toast.makeText(this@MainActivity, "CycleGAN error: ${t.message}", Toast.LENGTH_SHORT).show()
                        isProcessing = prevProcessingStateForCycle
                    } finally {
                        viewBinding.cycleGanButton.isEnabled = true
                    }
                }
            } else {
                // toggle OFF → cancel loop, hide overlay, resume pipeline
                cycleGanLoopJob?.cancel()
                cycleGanLoopJob = null

                viewBinding.processedFrameView.setImageBitmap(null)
                viewBinding.processedFrameView.visibility =
                    if (prevProcessingStateForCycle) View.VISIBLE else View.GONE
                isCycleGanApplied = false
                isProcessing = prevProcessingStateForCycle

                // reset button UI
                viewBinding.cycleGanButton.text = "CycleGAN"
                viewBinding.cycleGanButton.backgroundTintList =
                    ContextCompat.getColorStateList(this@MainActivity, R.color.green)

                Log.i("CycleGAN", "↩️ Restored live preview (processing=$isProcessing).")
            }
        }

        // Hook up pinch/slider zoom controls provided by CameraHelper/UI
        cameraHelper.setupZoomControls()
    }

    override fun onResume() {
        super.onResume()
        // Restart camera background thread and open camera (if the surface is ready)
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) {
            cameraHelper.openCamera()
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
        // Ensure the Take Photo visibility reflects current Settings when returning to the app
        if (shouldUpdateTakePhotoBtn) updateTakePhotoVisibility()
    }

    override fun onPause() {
        // If CycleGAN is showing, clear overlay & reset button first
        if (isCycleGanApplied) {
            isCycleGanApplied = false
            cycleGanOriginalBitmap = null
            viewBinding.processedFrameView.setImageBitmap(null)
            viewBinding.processedFrameView.visibility = View.GONE
            viewBinding.cycleGanButton.text = "CycleGAN"
            viewBinding.cycleGanButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
            Log.i("CycleGAN", "Paused: cleared CycleGAN overlay and reset button UI.")
        }

        // Pause live processing while backgrounded (even if not 'recording')
        isProcessing = false

        // Stop any ongoing recording first to flush/close encoders safely
        if (isRecording) stopProcessingAndRecording()

        // Release camera resources and background thread
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()

        super.onPause()
    }

    /* ------------------------------------------------------------------ */
    /*  Camera helpers                                                    */
    /* ------------------------------------------------------------------ */

    // Tracks which lens is active; CameraHelper reads this flag
    private var isFrontCamera = false

    /** Toggle camera facing and re-open the camera session. */
    private fun switchCamera() {
        if (isRecording) stopProcessingAndRecording() // stop processing before reconfiguring camera
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    /** Convenience check for both CAMERA and RECORD_AUDIO permissions. */
    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    /* ------------------------------------------------------------------ */
    /*  Start / Stop Processing                                           */
    /* ------------------------------------------------------------------ */

    /**
     * Enter "tracking" mode:
     *  - Set UI state and show the processed preview surface.
     *  - Reset the VideoProcessor (for any stateful internals).
     *  - If export is enabled, start a [ProcessedVideoRecorder] with model input dimensions.
     *
     * Note: We choose the model input (w,h) for the recorder. Adjust if your recorder expects 640×640.
     */
    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true

        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        videoProcessor?.reset()

        if (Settings.ExportData.videoDATA) {
            val dims = videoProcessor?.getModelDimensions()
            val w = dims?.first ?: 416
            val h = dims?.second ?: 416
            val path = ProcessedVideoRecorder.getExportedVideoOutputPath()
            processedVideoRecorder = ProcessedVideoRecorder(w, h, path).apply { start() }
        }
    }

    /**
     * Exit "tracking" mode:
     *  - Reset UI text/colors, hide and clear the processed preview,
     *  - Stop and release the recorder if it was active.
     */
    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false

        // clear any CycleGAN state when user stops
        isCycleGanApplied = false
        cycleGanOriginalBitmap = null
        viewBinding.processedFrameView.setImageBitmap(null)
        viewBinding.processedFrameView.visibility = View.GONE

        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        // clear any CycleGAN state when user stops
        isCycleGanApplied = false
        cycleGanOriginalBitmap = null

        // Reset CycleGAN button UI to default
        viewBinding.cycleGanButton.text = "CycleGAN"
        viewBinding.cycleGanButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.green)

        processedVideoRecorder?.stop()
        processedVideoRecorder = null
    }

    /* ------------------------------------------------------------------ */
    /*  Frame Processing                                                  */
    /* ------------------------------------------------------------------ */

    /**
     * Pull the latest preview bitmap from the TextureView and send it through [VideoProcessor].
     *
     * Guard rails:
     *  - `isProcessingFrame` prevents overlapping calls if preview updates faster than we process.
     *  - If processing was stopped while we were busy, we simply drop the result.
     *
     * Callback thread:
     *  - The current VideoProcessor implementation already posts the callback on the main thread.
     *    We still wrap with `runOnUiThread` here as a defensive measure; it’s harmless but redundant.
     */
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        videoProcessor?.processFrame(bitmap) { result ->
            runOnUiThread {
                if (!isProcessing) {            // user stopped meanwhile -> drop frame
                    isProcessingFrame = false
                    return@runOnUiThread
                }

                result?.let { (outBmp, preBmp) ->
                    // Show overlayed output on the ImageView
                    viewBinding.processedFrameView.setImageBitmap(outBmp)

                    // Optionally record the "plain" stretched frame (preBmp) to the video file
                    if (Settings.ExportData.videoDATA) {
                        processedVideoRecorder?.recordFrame(preBmp)
                    }
                } // if result is null: nothing to draw

                isProcessingFrame = false
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Model Loading (YOLO only)                                         */
    /* ------------------------------------------------------------------ */

    /**
     * Load a TFLite model from assets (copied into app files dir) on a background thread.
     *
     * Delegate strategy:
     *  - Tries NNAPI first (good for many Android devices).
     *  - Falls back to GPU delegate if NNAPI fails.
     *  - Falls back to pure CPU if both fail.
     *
     * The constructed Interpreter is passed to [VideoProcessor].
     */
    private fun loadTFLiteModelThreaded(assetName: String) {
        Thread {
            val path = copyAssetModel(assetName)
            if (path.isEmpty()) {
                runOnUiThread { Toast.makeText(this, "Failed to load $assetName", Toast.LENGTH_SHORT).show() }
                return@Thread
            }

            val options = Interpreter.Options().apply {
                setNumThreads(Runtime.getRuntime().availableProcessors())
                // Try NNAPI → GPU → CPU (no crash if unavailable)
                try { addDelegate(NnApiDelegate()) } catch (_: Exception) {
                    try { addDelegate(GpuDelegate()) } catch (_: Exception) { /* CPU fallback */ }
                }
            }

            yoloInterpreter = Interpreter(loadMappedFile(path), options)
            videoProcessor?.setInterpreter(yoloInterpreter!!)
        }.start()
    }

    /** Memory-map the TFLite file for faster loading (zero-copy-ish). */
    private fun loadMappedFile(modelPath: String): MappedByteBuffer =
        FileInputStream(File(modelPath)).channel.map(FileChannel.MapMode.READ_ONLY, 0, File(modelPath).length())

    /**
     * Copy a TFLite model from the app's assets into internal storage (if not already present).
     * Returns the absolute file path on success, or an empty string on failure.
     */
    private fun copyAssetModel(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buf = ByteArray(4 * 1024)
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) {
                        output.write(buf, 0, n)
                    }
                }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying $assetName: ${e.message}")
            ""
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Misc helpers                                                      */
    /* ------------------------------------------------------------------ */

    /** Show or hide the Take Photo button according to Settings; mark that we’ve updated it once. */
    private fun updateTakePhotoVisibility() {
        viewBinding.takePhotoButton.visibility =
            if (Settings.ExportData.takePhoto) View.VISIBLE else View.GONE
        shouldUpdateTakePhotoBtn = false
    }
}