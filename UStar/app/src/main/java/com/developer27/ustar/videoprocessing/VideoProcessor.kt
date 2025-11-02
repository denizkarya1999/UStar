@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
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

/* -----------------------------  Settings  ----------------------------- */

object Settings {

}
/** VideoProcessor */
class VideoProcessor(private val context: Context) {

    // Global variable to hold the processed bitmap
    private var processedBitmap: Bitmap? = null

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
    private suspend fun processFrameInternal(src: Bitmap): Bitmap =
        withContext(Dispatchers.IO) {
            // 1. Run CycleGAN inference first
            val cycleGanResult = runCycleGANInference(src)

            // 2. Run OpenCV processing and assign to global variable
            processedBitmap = runOpenCVProcessing(cycleGanResult)

            // 3. Classify the CycleGAN output
            currentPrediction = runResNet18Inference(processedBitmap!!)

            // 4. Return processed bitmap
            processedBitmap!!
        }

    /** Run CycleGAN inference on a given Bitmap */
    private fun runCycleGANInference(input: Bitmap): Bitmap {
        return try {
            com.developer27.ustar.machinelearning.CycleGAN.run(input)
        } catch (e: Exception) {
            // Show a toast instead of logging when inference fails
            Toast.makeText(context, "CycleGAN failed: ${e.message}", Toast.LENGTH_SHORT).show()
            input // fallback to original frame
        }
    }

    /** Run ResNet18 classifier on the bitmap produced by CycleGAN */
    private fun runResNet18Inference(input: Bitmap): String {
        // Load model (logs success/failure internally)
        val model = ResNet18.loadModel(context)
            ?: return "ResNet-18 Model Unavailable"

        val result = model.run(input)

        return result.topClass  // e.g., "cat"
    }

    /** TODO: <Ashwin Kumar Sarvadey> Placeholder for OpenCV-based image processing (will be implemented later) */
    private fun runOpenCVProcessing(bmp: Bitmap): Bitmap {
        // For now, just assign the unmodified bitmap to processedBitmap
        processedBitmap = bmp

        // TODO: Add OpenCV operations later and update processedBitmap accordingly
        return processedBitmap!!
    }
}
