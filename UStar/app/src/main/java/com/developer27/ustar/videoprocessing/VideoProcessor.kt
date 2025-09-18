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

/** Single detection in center-form coordinates. */
data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

/** Corner-form bounding box (x1,y1,x2,y2) for IoU/NMS or drawing. */
data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)

/* -----------------------------   global singletons   ----------------------------- */

/**
 * The active TFLite Interpreter used by this processor.
 * NOTE: A single Interpreter is typically *not* thread-safe across concurrent .run() calls.
 * If you process multiple frames in parallel, add synchronization or use one Interpreter per worker.
 */
private var tfliteInterpreter: Interpreter? = null

/** Final overlay canvas size used across the pipeline (stretched). */
private const val TARGET_OUT_W = 640
private const val TARGET_OUT_H = 640

/**
 * User-tunable settings grouped by purpose.
 * These are read in both VideoProcessor and YOLOHelper.
 */
object Settings {
    object DetectionMode {
        /** Toggle to enable/disable running YOLO inference (keeps overlay pipeline intact). */
        var enableYOLOinference: Boolean = true
    }
    object Inference {
        /**
         * Minimum confidence to accept a raw model output as a candidate.
         * Range: [0,1]. Example: 0.10f means “keep anything ≥ 10% confidence” *before* NMS.
         */
        var confidenceThreshold: Float = 0.10f

        /**
         * IoU threshold for NMS. Boxes that overlap a kept box by > IoU are discarded.
         * Range: [0,1]. Smaller = stricter deduplication.
         */
        var iouThreshold: Float = 0.10f
    }
    object BoundingBox {
        /** Whether to draw rectangles and labels on the overlay. */
        var enableBoundingBox: Boolean = true
        /** Box BGR color for OpenCV drawing. */
        var boxColor: Scalar = Scalar(0.0, 39.0, 76.0)  // BGR
        /** Rectangle stroke thickness in pixels. */
        var boxThickness: Int = 5
    }
    object ExportData {
        /** Example toggle for exporting frames/video to disk. Not used directly here. */
        var videoDATA: Boolean = false
        /** Example toggle to take a photo when a detection occurs. Not used directly here. */
        var takePhoto: Boolean = true
    }
}

/* -----------------------------      Processor       ----------------------------- */

/**
 * VideoProcessor
 * --------------
 * Orchestrates the end-to-end frame processing:
 *
 *   [Bitmap frame]
 *     → OpenCV init (once)
 *     → preprocess to 640×640 (stretch)
 *     → (optional) scale to model’s input size
 *     → TFLite inference (if enabled)
 *     → parse outputs + NMS
 *     → map detection to 640×640 space
 *     → draw overlays on an OpenCV Mat
 *     → return (overlayBitmap, plainBitmap), both 640×640
 *
 * Threading
 * ---------
 * - `processFrame(...)` launches work on Dispatchers.Default and does I/O bits on Dispatchers.IO.
 * - The Interpreter call happens on a background thread.
 * - Results are delivered on the main thread via the provided callback.
 *
 * Memory/Copies
 * -------------
 * This pipeline creates a few bitmaps per frame. For high-FPS scenarios, consider reusing:
 *  - a persistent Mat and Bitmap for the overlay,
 *  - avoiding repeated createScaledBitmap calls where possible.
 */
class VideoProcessor(private val context: Context) {

    init { initOpenCV() }

    /**
     * Load the OpenCV shared library. If this fails, drawing won’t work but the app survives.
     */
    private fun initOpenCV() {
        try { System.loadLibrary("opencv_java4") }
        catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    /**
     * Set (or replace) the active TFLite Interpreter used for inference.
     * We attempt to allocate tensors right away; if it fails, we still store the interpreter.
     */
    fun setInterpreter(model: Interpreter) {
        synchronized(this) {
            try { model.allocateTensors() } catch (_: Exception) {}
            tfliteInterpreter = model
        }
        Log.d("VideoProcessor", "TFLite model set successfully")
    }

    /** Simple user-facing reset stub; expand as needed. */
    fun reset() = Toast.makeText(context, "VideoProc Reset", Toast.LENGTH_SHORT).show()

    /**
     * Process a single frame asynchronously.
     *
     * @param bitmap   Input frame (arbitrary size).
     * @param callback Called on the main thread with:
     *                 - Pair.first  = overlay bitmap (boxes drawn) @ 640×640
     *                 - Pair.second = plain/stretched frame (no overlays) @ 640×640
     *                 or null on error.
     */
    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try { processFrameInternalYOLO(bitmap) } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    /**
     * The core pipeline for a single frame:
     *  - Determine model input/output shapes.
     *  - Preprocess to 640×640 (stretch).
     *  - Create a model-sized copy for inference.
     *  - Run TFLite, parse outputs, compute NMS.
     *  - Draw the best detection on a 640×640 Mat.
     *  - Return (overlay, plain) bitmaps, both 640×640.
     */
    private suspend fun processFrameInternalYOLO(src: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {
            val (inputW, inputH, outputShape) = getModelDimensions()

            // 0) Preprocess: Auto-orient + stretch to 640×640 (no original-size path)
            val preprocessed640 = YOLOHelper.preprocessInput(src) // returns 640×640

            // A) Inference bitmap: model-native size FROM the preprocessed frame
            val forModel = Bitmap.createScaledBitmap(preprocessed640, inputW, inputH, true)

            // B) Output “plain” image (already 640×640, no drawings)
            val stretched640 = preprocessed640

            // Prepare an OpenCV Mat at 640×640 that we can draw on
            val dstMat640 = Mat().also { Utils.bitmapToMat(preprocessed640, it) }

            val interpreter = tfliteInterpreter
            if (Settings.DetectionMode.enableYOLOinference && interpreter != null) {
                // Allocate an output buffer based on the discovered output shape.
                // Example shapes: [1, 5, N] or [1, N, 5].
                val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

                // Load bitmap into a TensorImage (FLOAT32). If you need normalization (e.g., /255f),
                // apply it when creating the ByteBuffer or preprocess the pixels beforehand.
                TensorImage(DataType.FLOAT32).apply { load(forModel) }
                    .also { interpreter.run(it.buffer, out) }

                // Parse outputs, do NMS, and get the best detection (if any).
                YOLOHelper.parseTFLite(out)?.let { det ->
                    // Map detection from model space to the 640×640 overlay space.
                    val rect640 = YOLOHelper.toTargetRect(
                        det = det,
                        targetW = TARGET_OUT_W, targetH = TARGET_OUT_H,
                        inputW = inputW, inputH = inputH,
                        boxesAreNormalized = true // set false if your model outputs absolute pixels
                    )
                    // Draw the box/label if enabled.
                    if (Settings.BoundingBox.enableBoundingBox) {
                        YOLOHelper.drawBoundingBoxes(dstMat640, rect640, det.confidence)
                    }
                }
            }

            // Convert the Mat (with drawings) back to a Bitmap.
            val processed640 = Bitmap.createBitmap(TARGET_OUT_W, TARGET_OUT_H, Bitmap.Config.ARGB_8888)
                .also { Utils.matToBitmap(dstMat640, it); dstMat640.release() }

            // Return: (overlay @ 640×640) to (plain @ 640×640)
            processed640 to stretched640
        }

    /**
     * Inspect the TFLite interpreter to learn input and output shapes.
     *
     * Returns:
     *   Triple(inputW, inputH, outputShapeList)
     * Where:
     *   - input tensor is assumed [1, H, W, C] → we read indices 1 and 2.
     *   - output tensor is assumed rank‑3 → e.g., [1, 5, N] or [1, N, 5].
     *
     * Fallbacks are provided if no interpreter is set:
     *   - input: 416×416
     *   - output: [1, 5, 3549]
     */
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor   = tfliteInterpreter?.getInputTensor(0)
        val shapeIn    = inTensor?.shape() // [1,H,W,C]
        val inputH     = shapeIn?.getOrNull(1) ?: 416
        val inputW     = shapeIn?.getOrNull(2) ?: 416

        val outTensor  = tfliteInterpreter?.getOutputTensor(0)
        val shapeOut   = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549) // [1,5,N] default

        require(shapeOut.size == 3) { "Model output must be rank-3 (got $shapeOut)" }
        return Triple(inputW, inputH, shapeOut)
    }
}