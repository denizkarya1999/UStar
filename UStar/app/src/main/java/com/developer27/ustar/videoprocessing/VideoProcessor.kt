@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import com.developer27.ustar.MainActivity
import com.developer27.ustar.MainActivity.Companion.currentPrediction
import com.developer27.ustar.machinelearning.ResNet18
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
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

            // 2. Run CycleGAN inference first
            val cycleGanResult = runCycleGANInference(src)

            // 3. Run OpenCV processing (starts the log content)
            processedBitmap = runOpenCVProcessing(cycleGanResult)

            // 4. Run ResNet-18 inference (appends prediction)
            currentPrediction = runResNet18Inference(processedBitmap!!)

            // 5. Write the full log to file (adds date + header)
            writeLogToFile()

            // 6. Return processed bitmap
            processedBitmap!!
        }

    /** Run CycleGAN inference on a given Bitmap */
    private fun runCycleGANInference(input: Bitmap): Bitmap {
        return try {
            com.developer27.ustar.machinelearning.CycleGAN.run(input)
        } catch (e: Exception) {
            Toast.makeText(context, "CycleGAN failed: ${e.message}", Toast.LENGTH_SHORT).show()
            input
        }
    }

    /** Run ResNet-18 classifier and append its result to the global log */
    private fun runResNet18Inference(input: Bitmap): String {
        val model = ResNet18.loadModel(context)
            ?: return "ResNet-18 Model Unavailable"

        val result = model.run(input)
        val prediction = result.topClass

        // Append the ResNet result line
        logMessage.appendLine("ResNet-18 Prediction: $prediction")

        return prediction
    }

    /** OpenCV-based image processing */
    // TODO: <Ashwin Kumar Sarvadey> After CycleGAN is implemented, do the necessary image processings and log predictions based on the paper.
    private fun runOpenCVProcessing(bmp: Bitmap): Bitmap {
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
