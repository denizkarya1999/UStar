package com.developer27.ustar

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
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
import com.developer27.ustar.camera.CameraHelper
import com.developer27.ustar.databinding.ActivityMainBinding
import com.developer27.ustar.videoprocessing.ProcessedVideoRecorder
import com.developer27.ustar.videoprocessing.Settings
import com.developer27.ustar.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var yoloInterpreter: Interpreter? = null

    private var processedVideoRecorder: ProcessedVideoRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false
    private var shouldUpdateTakePhotoBtn = true

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, w: Int, h: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
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
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        cameraHelper   = CameraHelper(this, viewBinding)
        videoProcessor = VideoProcessor(this)

        // hide processed‑frame preview initially
        viewBinding.processedFrameView.visibility = View.GONE

        /*------ app / web link ------*/
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.zhangxiao.me/")))
        }

        /*------ permission launcher ------*/
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

        /*------ UI buttons ------*/
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording() else startProcessingAndRecording()
        }
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutUStarActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        viewBinding.takePhotoButton.setOnClickListener {
            cameraHelper.takePhoto { file ->
                Toast.makeText(this, "Photo saved: ${file.absolutePath}", Toast.LENGTH_SHORT).show()
                val bmp = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
                viewBinding.processedFrameView.setImageBitmap(bmp)
            }
        }

        // show / hide Take Photo button based on Settings
        updateTakePhotoVisibility()

        /*------ load YOLO model only ------*/
        loadTFLiteModelThreaded("YOLOv3_float32.tflite")

        cameraHelper.setupZoomControls()
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) {
            cameraHelper.openCamera()
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
        if (shouldUpdateTakePhotoBtn) updateTakePhotoVisibility()
    }

    override fun onPause() {
        if (isRecording) stopProcessingAndRecording()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    /* ------------------------------------------------------------------ */
    /*  Camera helpers                                                    */
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
    /*  Start / Stop Processing                                           */
    /* ------------------------------------------------------------------ */
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

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        processedVideoRecorder?.stop()
        processedVideoRecorder = null
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
                if (!isProcessing) {            // stopped meanwhile
                    isProcessingFrame = false
                    return@runOnUiThread
                }

                result?.let { (outBmp, preBmp) ->
                    viewBinding.processedFrameView.setImageBitmap(outBmp)
                    if (Settings.ExportData.videoDATA) {
                        processedVideoRecorder?.recordFrame(preBmp)
                    }
                }                                   // if result is null: nothing to draw

                isProcessingFrame = false
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Model Loading (YOLO only)                                         */
    /* ------------------------------------------------------------------ */
    private fun loadTFLiteModelThreaded(assetName: String) {
        Thread {
            val path = copyAssetModel(assetName)
            if (path.isEmpty()) {
                runOnUiThread { Toast.makeText(this, "Failed to load $assetName", Toast.LENGTH_SHORT).show() }
                return@Thread
            }
            val options = Interpreter.Options().apply {
                setNumThreads(Runtime.getRuntime().availableProcessors())
                try { addDelegate(NnApiDelegate()) } catch (_: Exception) {
                    try { addDelegate(GpuDelegate()) } catch (_: Exception) { /* CPU fallback */ }
                }
            }
            yoloInterpreter = Interpreter(loadMappedFile(path), options)
            videoProcessor?.setInterpreter(yoloInterpreter!!)
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer =
        FileInputStream(File(modelPath)).channel.map(FileChannel.MapMode.READ_ONLY, 0, File(modelPath).length())

    private fun copyAssetModel(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buf = ByteArray(4 * 1024)
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) output.write(buf, 0, n)
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
    private fun updateTakePhotoVisibility() {
        viewBinding.takePhotoButton.visibility =
            if (Settings.ExportData.takePhoto) View.VISIBLE else View.GONE
        shouldUpdateTakePhotoBtn = false
    }
}
