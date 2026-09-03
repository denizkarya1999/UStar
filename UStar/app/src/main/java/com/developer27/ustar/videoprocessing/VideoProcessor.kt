@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import com.developer27.ustar.MainActivity.Companion.currentPrediction
import com.developer27.ustar.machinelearning.DynaSpa.MiniDynaSpaPreprocessor
import com.developer27.ustar.machinelearning.Orientation_Guidance_ResNet18
import com.developer27.ustar.machinelearning.Optical_Ranging_ResNet18
import com.developer27.ustar.storage.PredictionLogWriter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/* -----------------------------  Settings  ----------------------------- */
object Settings {
    var selectedDenoiser: String = "cyclegan"
}

/** VideoProcessor */
class VideoProcessor(context: Context) {

    private val context = context.applicationContext
    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // Global variable to hold the processed bitmap
    private var processedBitmap: Bitmap? = null

    // Global variable for combined log message
    private var logMessage: StringBuilder = StringBuilder()

    /** Simple reset placeholder. */
    fun reset() = Toast.makeText(context, "Video Processor Reset", Toast.LENGTH_SHORT).show()

    /** Process a single frame asynchronously. */
    fun processFrame(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        processorScope.launch {
            val result = try {
                processFrameInternal(bitmap)
            } catch (e: Exception) {
                Log.e("VideoProcessor", "Error processing frame", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    fun close() {
        processorScope.cancel()
    }

    /** Process the frame with deep learning models */
    private fun processFrameInternal(src: Bitmap): Bitmap {
        // 1. Reset the logMessage variable for logging
        logMessage = StringBuilder()

        // 2. Run the selected denoising model first
        val denoisedResult = when (Settings.selectedDenoiser.lowercase(Locale.ROOT)) {
            "dynaspa" -> runDynaSpaDenoisingInference(src)
            else -> runCycleGANDenoisingInference(src)
        }

        // 3. Run ResNet-18 inference for orientation and optical ranging
        currentPrediction = runResNet18CombinedInference(denoisedResult)

        // 4. Write the full log to file
        writeLogToFile()

        // 5. Return processed bitmap
        return denoisedResult
    }

    /** Run tag detection model and return its DynaSpa masked result as a Bitmap */
    private fun runDynaSpaDenoisingInference(input: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()
        val resultBitmap = try {
            val result = MiniDynaSpaPreprocessor.run(context, input)

            if (result != null) {
                val box = MiniDynaSpaPreprocessor.extractBoundingBoxFromFeatureMap(
                    featureMapTensor = result.featureMap,
                    outputWidth = result.processedBitmap.width,
                    outputHeight = result.processedBitmap.height
                )

                if (MiniDynaSpaPreprocessor.shouldShowBoundingBox(result.predictedClass)) {
                    com.developer27.ustar.machinelearning.DynaSpa.DynaSpaMaskProcessor.processBoundingBox(
                        source = result.processedBitmap,
                        boundingBox = box
                    )
                } else {
                    Bitmap.createBitmap(
                        result.processedBitmap.width,
                        result.processedBitmap.height,
                        Bitmap.Config.ARGB_8888
                    ).apply { eraseColor(android.graphics.Color.BLACK) }
                }
            } else {
                input
            }
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Tag detection DynaSpa masking failed", e)
            input
        }
        val endTime = System.currentTimeMillis()
        logMessage.appendLine("DynaSpa Denoising Inference Time: ${endTime - startTime} ms")
        return resultBitmap
    }

    /** Run CycleGAN denoising inference on a given Bitmap */
    private fun runCycleGANDenoisingInference(input: Bitmap): Bitmap {
        val startTime = System.currentTimeMillis()
        val resultBitmap = try {
            com.developer27.ustar.machinelearning.Denoising_CycleGAN.load(context)
            com.developer27.ustar.machinelearning.Denoising_CycleGAN.run(input)
        } catch (e: Exception) {
            Log.e("VideoProcessor", "CycleGAN based denoising failed", e)
            input
        }
        val endTime = System.currentTimeMillis()
        logMessage.appendLine("CycleGAN Denoising Inference Time: ${endTime - startTime} ms")
        return resultBitmap
    }

    /** Run ResNet-18 classifier for orientation and optical ranging and append its result to the global log */
    private fun runResNet18CombinedInference(input: Bitmap): String {
        // Run Orientation model
        val orientationModel = Orientation_Guidance_ResNet18.loadModel(context)
            ?: return "ResNet-18 based Orientation Model Unavailable"

        // Run Optical Ranging model
        val opticalRangingModel = Optical_Ranging_ResNet18.loadModel(context)
            ?: return "ResNet-18 based Optical Ranging Model Unavailable"

        // Run inference for optical ranging
        val rangingStartTime = System.currentTimeMillis()
        val rangingResult = opticalRangingModel.run(input)
        val rangingEndTime = System.currentTimeMillis()
        val rangingPrediction = rangingResult.topClass

        // Run inference for orientation
        val orientationStartTime = System.currentTimeMillis()
        val orientationResult = orientationModel.run(input)
        val orientationEndTime = System.currentTimeMillis()
        val orientationPrediction = orientationResult.topClass

        // Combined prediction
        val prediction = "Distance: $rangingPrediction | Orientation: $orientationPrediction"

        // Append the ResNet result line
        logMessage.appendLine(prediction)
        logMessage.appendLine("Optical Ranging Inference Time: ${rangingEndTime - rangingStartTime} ms")
        logMessage.appendLine("Orientation Guidance Inference Time: ${orientationEndTime - orientationStartTime} ms")

        return prediction
    }

    /** Writes the full log with date and header to Documents/UStar_Cube_Prediction.txt */
    private fun writeLogToFile() {
        PredictionLogWriter.write(context, logMessage)
    }
}
