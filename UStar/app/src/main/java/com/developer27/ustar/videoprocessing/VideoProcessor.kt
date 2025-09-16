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
import org.opencv.core.Scalar
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage

/* -----------------------------  simple data classes  ----------------------------- */

data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)

/* -----------------------------   global singletons   ----------------------------- */
private var tfliteInterpreter: Interpreter? = null
private const val TARGET_OUT_W = 640
private const val TARGET_OUT_H = 640

object Settings {
    object DetectionMode {
        var enableYOLOinference: Boolean = true
    }
    object Inference {
        var confidenceThreshold: Float = 0.10f
        var iouThreshold: Float = 0.10f
    }
    object BoundingBox {
        var enableBoundingBox: Boolean = true
        var boxColor: Scalar = Scalar(0.0, 39.0, 76.0)  // BGR
        var boxThickness: Int = 5
    }
    object ExportData {
        var videoDATA: Boolean = false
        var takePhoto: Boolean = true
    }
}

/* -----------------------------      Processor       ----------------------------- */

class VideoProcessor(private val context: Context) {

    init { initOpenCV() }

    private fun initOpenCV() {
        try { System.loadLibrary("opencv_java4") }
        catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    fun setInterpreter(model: Interpreter) {
        synchronized(this) {
            try { model.allocateTensors() } catch (_: Exception) {}
            tfliteInterpreter = model
        }
        Log.d("VideoProcessor", "TFLite model set successfully")
    }

    fun reset() = Toast.makeText(context, "VideoProc Reset", Toast.LENGTH_SHORT).show()

    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try { processFrameInternalYOLO(bitmap) } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    private suspend fun processFrameInternalYOLO(src: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {
            val (inputW, inputH, outputShape) = getModelDimensions()
            logDimsOnce(inputW, inputH, outputShape)

            // 0) Preprocess: Auto-orient + stretch to 640×640 (no original-size path)
            val preprocessed640 = YOLOHelper.preprocessInput(src) // returns 640×640

            // A) Inference bitmap: model-native size FROM the preprocessed frame
            val forModel = Bitmap.createScaledBitmap(preprocessed640, inputW, inputH, true)

            // B) Output “plain” image (already 640×640)
            val stretched640 = preprocessed640

            // Work directly at 640×640 for overlays
            val dstMat640 = Mat().also { Utils.bitmapToMat(preprocessed640, it) }

            val interpreter = tfliteInterpreter
            if (Settings.DetectionMode.enableYOLOinference && interpreter != null) {
                val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

                TensorImage(DataType.FLOAT32).apply { load(forModel) }
                    .also { interpreter.run(it.buffer, out) }

                YOLOHelper.parseTFLite(out)?.let { det ->
                    // Map detections directly to 640×640
                    val rect640 = YOLOHelper.toTargetRect(
                        det = det,
                        targetW = TARGET_OUT_W, targetH = TARGET_OUT_H,
                        inputW = inputW, inputH = inputH,
                        boxesAreNormalized = true
                    )
                    if (Settings.BoundingBox.enableBoundingBox) {
                        YOLOHelper.drawBoundingBoxes(dstMat640, rect640, det.confidence)
                    }
                }
            }

            val processed640 = Bitmap.createBitmap(TARGET_OUT_W, TARGET_OUT_H, Bitmap.Config.ARGB_8888)
                .also { Utils.matToBitmap(dstMat640, it); dstMat640.release() }

            // Return: (overlay @ 640×640) to (plain @ 640×640)
            processed640 to stretched640
        }

    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor   = tfliteInterpreter?.getInputTensor(0)
        val shapeIn    = inTensor?.shape() // [1,H,W,C]
        val inputH     = shapeIn?.getOrNull(1) ?: 416
        val inputW     = shapeIn?.getOrNull(2) ?: 416
        val outTensor  = tfliteInterpreter?.getOutputTensor(0)
        val shapeOut   = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549) // [1,5,N]
        require(shapeOut.size == 3) { "Model output must be rank-3 (got $shapeOut)" }
        return Triple(inputW, inputH, shapeOut)
    }

    private var loggedDims = false
    private fun logDimsOnce(inW: Int, inH: Int, outShape: List<Int>) {
        if (!loggedDims) {
            Log.d("VideoProcessor", "Model input: ${inW}x${inH}, output shape: $outShape")
            loggedDims = true
        }
    }
}