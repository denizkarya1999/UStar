@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.developer27.ustar.MainActivity.Companion.currentPrediction
import com.developer27.ustar.machinelearning.Orientation_ResNet18
import com.developer27.ustar.machinelearning.Optical_Ranging_ResNet18
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/* -----------------------------  Settings  ----------------------------- */
object Settings {
}

/** VideoProcessor */
class VideoProcessor(private val context: Context) {

    // Global variable to hold the processed bitmap
    private var processedBitmap: Bitmap? = null

    // Global variable to store second preprocessed image for Orientation Guidance and Optical Ranging Models
    private var secondPreprocessedImage: Bitmap? = null

    // Global variable for combined log message
    private var logMessage: StringBuilder = StringBuilder()

    init { initOpenCV() }

    /** Load the OpenCV shared library. */
    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    /** Simple reset placeholder. */
    fun reset() = Toast.makeText(context, "Video Processor Reset", Toast.LENGTH_SHORT).show()

    /** Process a single frame asynchronously. */
    fun processFrame(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try {
                processFrameInternal(bitmap)
            } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    /** Process the frame with computer vision and other deep learning models */
    private suspend fun processFrameInternal(src: Bitmap): Bitmap = withContext(Dispatchers.IO) {
            // 1. Reset the logMessage variable for logging
            logMessage = StringBuilder()

            // 2. Run CycleGAN based denoising inference first
            val cycleGanResult = runCycleGANDenoisingInference(src)

            // 3. Run OpenCV processing for Computer Vision algorithms
            processedBitmap = runCVProcessing(cycleGanResult)

            // 4. Increase image padding to 789x355 to mimic training dataset dimensions for Orientation Guidance and Optical Ranging Models
            secondPreprocessedImage = increaseImagePadding(src = processedBitmap!!, outW = 789, outH = 355)

            // 5. Run ResNet-18 inference for orientation and optical ranging (using noised image for now)
            currentPrediction = runResNet18CombinedInference(secondPreprocessedImage!!)

            // 6. Write the full log to file (adds date + header)
            writeLogToFile()

            // 7. Return processed bitmap
            processedBitmap!!
    }

    /** Run CycleGAN denoising inference on a given Bitmap */
    private fun runCycleGANDenoisingInference(input: Bitmap): Bitmap {
        return try {
            com.developer27.ustar.machinelearning.Denoising_CycleGAN.run(input)
        } catch (e: Exception) {
            Toast.makeText(context, "CycleGAN based denoising failed: ${e.message}", Toast.LENGTH_SHORT).show()
            input
        }
    }

    /** Run ResNet-18 classifier for orientation and optical ranging and append its result to the global log */
    private fun runResNet18CombinedInference(input: Bitmap): String {
        // Run Orientation model
        val orientationModel = Orientation_ResNet18.loadModel(context)
            ?: return "ResNet-18 based Orientation Model Unavailable"

        // Run Optical Ranging model
        val opticalRangingModel = Optical_Ranging_ResNet18.loadModel(context)
            ?: return "ResNet-18 based Optical Ranging Model Unavailable"

        // Run inference for optical ranging
        val rangingResult = opticalRangingModel.run(input)
        val rangingPrediction = rangingResult.topClass

        // Run inference for orientation
        val orientationResult = orientationModel.run(input)
        val orientationPrediction = orientationResult.topClass

        // Combined prediction
        val prediction = "Distance: $rangingPrediction | Orientation: $orientationPrediction"

        // Append the ResNet result line
        logMessage.appendLine(prediction)

        return prediction
    }

    /** CV-based image processing */
    // TODO: <Ashwin Kumar Sarvadey> After CycleGAN is implemented, do the necessary image processings and log predictions based on the paper.
    private fun runCVProcessing(bmp: Bitmap): Bitmap {
        // Initialize the bitmap
        processedBitmap = bmp

        // Check if OpenCV is initialized by trying to create a Mat object
        val isInitialized = try {
            val testMat = Mat()
            testMat.release()
            true
        } catch (e: Exception) {
            false
        }

        // Log the initialization status.
        logMessage.appendLine("OpenCV Initialization Status: $isInitialized")

        // TODO: <Ashwin Kumar Sarvadey> Perform necessary processing to processedBitmap and log features based on the map to LogMessage variable.

        // Placeholder for feature logging
        logMessage.appendLine("Test feature is logged in by OpenCV")

        return processedBitmap!!
    }

    // Image resizing for Optical Ranging and Orientation Guidance Model Inference
    private fun increaseImagePadding(
        src: Bitmap,
        outW: Int,
        outH: Int
    ): Bitmap {

        // ---- 1) Create the output canvas (fixed size) ----
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        // ---- 2) Fill the whole background with pure black ----
        canvas.drawARGB(255, 0, 0, 0)

        // ---- 3) Center the ORIGINAL image ----
        val left = (outW - src.width) / 2f
        val top  = (outH - src.height) / 2f

        canvas.drawBitmap(src, left, top, null)

        return output
    }

    /** Writes the full log with date and header to Documents/UStar_Cube_Prediction.txt */
    private fun writeLogToFile() {
        try {
            val documentsDir =
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (!documentsDir.exists()) documentsDir.mkdirs()

            val logFile = File(documentsDir, "UStar_Cube_Prediction.txt")

            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

            // Build the final output
            val fullLog = StringBuilder()
                .appendLine("UStar UIOD Tag Features")
                .appendLine("Prediction Date: $timestamp")
                .append(logMessage.toString())

            // Overwrite the file
            FileWriter(logFile, false).use { writer ->
                writer.write(fullLog.toString())
            }

            Log.i("UStarLogger", "File overwritten with:\n$fullLog")
        } catch (e: Exception) {
            Log.e("UStarLogger", "Error writing log file: ${e.message}")
        }
    }
}
