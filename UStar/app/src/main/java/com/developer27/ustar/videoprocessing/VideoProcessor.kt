@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Mat

/* -----------------------------  Settings  ----------------------------- */

object Settings {

}
/** VideoProcessor */
class VideoProcessor(private val context: Context) {

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

            // 2. Then run OpenCV processing later (placeholder for now)
            runOpenCVProcessing(cycleGanResult)
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

    /** TODO: <Ashwin Kumar Sarvadey> Placeholder for OpenCV-based image processing (will be implemented later) */
    private fun runOpenCVProcessing(input: Bitmap): Bitmap {
        // TODO: Add OpenCV operations after CycleGAN output
        return input
    }
}
