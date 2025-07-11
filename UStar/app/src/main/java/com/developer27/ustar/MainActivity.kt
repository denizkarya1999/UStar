package com.developer27.ustar

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
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
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.ustar.camera.CameraHelper
import com.developer27.ustar.databinding.ActivityMainBinding
import com.developer27.ustar.videoprocessing.ProcessedFrameRecorder
import com.developer27.ustar.videoprocessing.ProcessedVideoRecorder
import com.developer27.ustar.videoprocessing.Settings
import com.developer27.ustar.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var tfliteInterpreter: Interpreter? = null
    // Interpreter for letter recognition.
    private var letterInterpreter: Interpreter? = null

    private var processedVideoRecorder: ProcessedVideoRecorder? = null
    private var processedFrameRecorder: ProcessedFrameRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    // Flag for tracking (start/stop tracking mode)
    private var isRecording = false
    // Flag for frame processing
    private var isProcessing = false
    private var isProcessingFrame = false

    // Variable for inference result.
    private var inferenceResult = ""

    // Stores the current session tracking coordinates.
    private var trackingCoordinates: String = ""

    // For toggling digit/letter recognition.
    var isLetterSelected = true
    var isDigitSelected = !isLetterSelected

    // Flag for writing mode.
    private var isWriting = false

    // Flag to clear prediction when returning from an external intent.
    private var shouldClearPrediction = false

    // NEW: Accumulated handwriting coordinates (each element corresponds to one letter)
    private val accumulatedCoordinates = mutableListOf<String>()

    // NEW: Flag to indicate a reset is happening, to guard against asynchronous updates.
    private var isResetting = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

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

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"

        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val camGranted = permissions[Manifest.permission.CAMERA] ?: false
            val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
            if (camGranted && micGranted) {
                if (viewBinding.viewFinder.isAvailable) {
                    cameraHelper.openCamera()
                } else {
                    viewBinding.viewFinder.surfaceTextureListener = textureListener
                }
            } else {
                Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
            }
        }

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutUStarActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Set up the letter/digit switch.
        val letterDigitSwitch = viewBinding.letterDigitSwitch
        if (isLetterSelected) {
            letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFCB05"))
            letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFCB05")
            )
            letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFCB05")
            )
            letterDigitSwitch.text = "Letter"
        } else {
            letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFFFFF"))
            letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFFFFF")
            )
            letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                android.graphics.Color.parseColor("#FFFFFF")
            )
            letterDigitSwitch.text = "Digit"
        }
        letterDigitSwitch.isChecked = isLetterSelected

        letterDigitSwitch.setOnCheckedChangeListener { _, isChecked ->
            isLetterSelected = isChecked
            isDigitSelected = !isChecked
            if (isChecked) {
                letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFCB05"))
                letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFCB05")
                )
                letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFCB05")
                )
                letterDigitSwitch.text = "Letter"
            } else {
                letterDigitSwitch.setTextColor(android.graphics.Color.parseColor("#FFFFFF"))
                letterDigitSwitch.thumbTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFFFFF")
                )
                letterDigitSwitch.trackTintList = android.content.res.ColorStateList.valueOf(
                    android.graphics.Color.parseColor("#FFFFFF")
                )
                letterDigitSwitch.text = "Digit"
            }
        }

        // Set up the Start Writing button.
        viewBinding.startWritingButton.setOnClickListener {
            toggleWritingMode()
        }

        // Set up the Clear Prediction button ("C").
        viewBinding.clearPredictionButton.setOnClickListener {
            if (isWriting) {
                isWriting = false
                viewBinding.startWritingButton.text = "Start Writing"
                viewBinding.startWritingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.green)
            }
            if (isRecording) {
                stopProcessingAndRecording()
            }
            // Reset prediction text and stored coordinates while guarding against asynchronous updates.
            isResetting = true
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
            isResetting = false
        }

        // Load ML models.
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")
        loadTFLiteModelOnStartupThreaded("DigitRecog_float32.tflite")
        loadTFLiteModelOnStartupThreaded("LetterRecog_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }
    }

    // Function for toggling writing mode.
    private fun toggleWritingMode() {
        if (!isWriting) {
            // Reset any previously written content while setting a guard flag.
            isResetting = true
            viewBinding.predictedLetterTextView.text = ""
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
            isResetting = false

            isWriting = true
            viewBinding.startWritingButton.text = "Stop Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.red)
        } else {
            isWriting = false
            viewBinding.startWritingButton.text = "Start Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
            val prediction = viewBinding.predictedLetterTextView.text.toString()
            if (prediction.matches(Regex("^(?=.*[A-Za-z])(?=.*\\d).+$"))) {
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ -> sendEmail(prediction) }
                    .setNegativeButton("No") { _, _ -> launch3DActivity() }
                    .show()
            } else if (isLetterSelected) {
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ -> sendEmail(prediction) }
                    .setNegativeButton("No") { _, _ -> launch3DActivity() }
                    .show()
            } else {
                if (prediction.matches(Regex("\\d+"))) {
                    AlertDialog.Builder(this)
                        .setTitle("Call Number")
                        .setMessage("Do you wish to call the number $prediction?")
                        .setPositiveButton("Yes") { _, _ -> makePhoneCall(prediction) }
                        .setNegativeButton("No") { _, _ -> launch3DActivity() }
                        .show()
                } else {
                    launch3DActivity()
                }
            }
        }
    }

    // New function to send an email with the prediction.
    private fun sendEmail(text: String) {
        val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_SUBJECT, "Air-Written Email by Xamera")
            putExtra(Intent.EXTRA_TEXT, text)
        }
        shouldClearPrediction = true
        startActivity(emailIntent)
    }

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
            val width = dims?.first ?: 416
            val height = dims?.second ?: 416
            val outputPath = ProcessedVideoRecorder.getExportedVideoOutputPath()
            processedVideoRecorder = ProcessedVideoRecorder(width, height, outputPath)
            processedVideoRecorder?.start()
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

        val outputPath = get28x28OutputPath()
        processedFrameRecorder = ProcessedFrameRecorder(outputPath)
        with(Settings.ExportData) {
            if (frameIMG) {
                val bitmap = videoProcessor?.exportTraceForInference()
                if (bitmap != null) {
                    processedFrameRecorder?.save(bitmap)
                }
            }
        }

        initializeInferenceResult()

        if (isWriting) {
            val currentCoords = videoProcessor?.getTrackingCoordinatesString() ?: ""
            if (currentCoords.isNotEmpty()) {
                accumulateCoordinates(currentCoords)
            }
            val currentText = viewBinding.predictedLetterTextView.text.toString()
            val newText = if (currentText == "No Prediction Available Yet") {
                inferenceResult
            } else {
                currentText + inferenceResult
            }
            viewBinding.predictedLetterTextView.text = newText
        } else {
            viewBinding.predictedLetterTextView.text = inferenceResult
        }

        trackingCoordinates = videoProcessor?.getTrackingCoordinatesString() ?: ""
    }

    // NEW: Function to accumulate and horizontally offset new tracking coordinates.
    private fun accumulateCoordinates(newCoords: String) {
        if (newCoords.isEmpty()) return
        if (accumulatedCoordinates.isEmpty()) {
            accumulatedCoordinates.add(newCoords)
        } else {
            var offsetX = 0.0
            for (coordStr in accumulatedCoordinates) {
                val pts = coordStr.split(";").mapNotNull {
                    val parts = it.split(",")
                    parts.getOrNull(0)?.toDoubleOrNull()
                }
                if (pts.isNotEmpty()) {
                    val currentMax = pts.maxOrNull() ?: 0.0
                    offsetX = max(offsetX, currentMax)
                }
            }
            offsetX += 10.0
            val adjustedPoints = newCoords.split(";").mapNotNull { pointStr ->
                val parts = pointStr.split(",")
                if (parts.size >= 2) {
                    val x = parts[0].toDoubleOrNull() ?: 0.0
                    val y = parts[1]
                    val z = if (parts.size >= 3) parts[2] else "0.0"
                    "${(x + offsetX)},$y,$z"
                } else null
            }
            val adjustedCoords = adjustedPoints.joinToString(separator = ";")
            accumulatedCoordinates.add(adjustedCoords)
        }
    }

    private fun get28x28OutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val rollityDir = File(picturesDir, "Exported Lines from Xamera")
        if (!rollityDir.exists()) {
            rollityDir.mkdirs()
        }
        return File(rollityDir, "DrawnLine_28x28_${System.currentTimeMillis()}.png").absolutePath
    }

    private fun initializeInferenceResult() {
        if (isLetterSelected) {
            inferenceResult = runLetterRecognitionInference()
        } else if (isDigitSelected) {
            inferenceResult = runDigitRecognitionInference()
        }
    }

    private fun runDigitRecognitionInference(): String {
        val digitBitmap = videoProcessor?.exportTraceForInference()
        if (digitBitmap == null) {
            Log.e("MainActivity", "No digit image available for inference")
            return "Error"
        }
        val grayBitmap = convertToGrayscale(digitBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        val outputArray = Array(1) { FloatArray(10) }
        if (tfliteInterpreter == null) {
            Log.e("MainActivity", "Digit model interpreter not set")
            return "Error"
        }
        tfliteInterpreter?.run(inputBuffer, outputArray)
        val predictedDigit = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        Log.d("MainActivity", "Digit model predicted: $predictedDigit")
        return predictedDigit.toString()
    }

    private fun runLetterRecognitionInference(): String {
        val letterBitmap = videoProcessor?.exportTraceForInference()
        if (letterBitmap == null) {
            Log.e("MainActivity", "No letter image available for inference")
            return "Error"
        }
        val grayBitmap = convertToGrayscale(letterBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        val outputArray = Array(1) { FloatArray(26) }
        if (letterInterpreter == null) {
            Log.e("MainActivity", "Letter model interpreter not set")
            return "Error"
        }
        letterInterpreter?.run(inputBuffer, outputArray)
        val maxIndex = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        if (maxIndex == -1) {
            return "Error"
        }
        val predictedLetter = ('A'.toInt() + maxIndex).toChar()
        Log.d("MainActivity", "Letter model predicted: $predictedLetter")
        return predictedLetter.toString()
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    private fun convertBitmapToGrayscaleByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputSize = bitmap.width * bitmap.height
        val byteBuffer = ByteBuffer.allocateDirect(inputSize * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF).toFloat()
            val normalized = r / 255.0f
            byteBuffer.putFloat(normalized)
        }
        return byteBuffer
    }

    // Modified launch3DActivity(): Combine accumulated coordinates.
    private fun launch3DActivity() {
        val coords = if (accumulatedCoordinates.isNotEmpty())
            accumulatedCoordinates.joinToString(separator = "|")
        else if (trackingCoordinates.isNotEmpty())
            trackingCoordinates
        else
            "0.0,0.0,0.0;5.0,10.0,-5.0;-5.0,15.0,10.0;20.0,-5.0,5.0;-10.0,0.0,-10.0;10.0,-15.0,15.0;0.0,20.0,-5.0"
        val intent = Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java)
        intent.putExtra("LETTER_KEY", viewBinding.predictedLetterTextView.text.toString())
        intent.putExtra("PATH_COORDINATES", coords)
        shouldClearPrediction = true
        startActivity(intent)
    }

    // Simulate making a phone call using ACTION_DIAL.
    private fun makePhoneCall(digits: String) {
        val callIntent = Intent(Intent.ACTION_DIAL)
        callIntent.data = Uri.parse("tel:$digits")
        shouldClearPrediction = true
        startActivity(callIntent)
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                // If a reset is in progress, skip updating the UI.
                if (isResetting) {
                    isProcessingFrame = false
                    return@runOnUiThread
                }
                processedFrames?.let { (outputBitmap, preprocessedBitmap) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                        with(Settings.ExportData) {
                            if (videoDATA) {
                                processedVideoRecorder?.recordFrame(preprocessedBitmap)
                            }
                        }
                    }
                }
                if (videoProcessor?.getTrackingCoordinatesString().isNullOrEmpty()) {
                    resetScreen()
                }
                isProcessingFrame = false
            }
        }
    }

    private fun resetScreen() {
        // Use the guard flag while resetting.
        isResetting = true
        viewBinding.processedFrameView.setImageBitmap(null)
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"
        trackingCoordinates = ""
        isResetting = false
    }

    private fun getProcessedVideoOutputPath(): String {
        @Suppress("DEPRECATION")
        val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
        if (!moviesDir.exists()) {
            moviesDir.mkdirs()
        }
        return File(moviesDir, "Processed_${System.currentTimeMillis()}.mp4").absolutePath
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(modelName)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val options = Interpreter.Options().apply {
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        var delegateAdded = false
                        try {
                            val nnApiDelegate = NnApiDelegate()
                            options.addDelegate(nnApiDelegate)
                            delegateAdded = true
                            Log.d("MainActivity", "NNAPI delegate added successfully.")
                        } catch (e: Exception) {
                            Log.d("MainActivity", "NNAPI delegate unavailable, falling back to GPU delegate.", e)
                        }
                        if (!delegateAdded) {
                            try {
                                val gpuDelegate = GpuDelegate()
                                options.addDelegate(gpuDelegate)
                                Log.d("MainActivity", "GPU delegate added successfully.")
                            } catch (e: Exception) {
                                Log.d("MainActivity", "GPU delegate unavailable, will use CPU only.", e)
                            }
                        }
                        when (modelName) {
                            "YOLOv3_float32.tflite" -> {
                                videoProcessor?.setInterpreter(Interpreter(loadMappedFile(bestLoadedPath), options))
                            }
                            "DigitRecog_float32.tflite" -> {
                                tfliteInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                            }
                            "LetterRecog_float32.tflite" -> {
                                letterInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                            }
                            else -> Log.d("MainActivity", "No model processing method defined for $modelName")
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.d("MainActivity", "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile.absolutePath
            }
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                    }
                    output.flush()
                }
            }
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
            ""
        }
    }

    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    override fun onResume() {
        super.onResume()
        // Clear accumulated handwriting coordinates when returning.
        accumulatedCoordinates.clear()

        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
        if (shouldClearPrediction) {
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            shouldClearPrediction = false
        }
    }

    override fun onPause() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}