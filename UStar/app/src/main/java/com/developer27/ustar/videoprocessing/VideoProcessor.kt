@file:Suppress("SameParameterValue")

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
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.max
import kotlin.math.min

/* -----------------------------  simple data classes  ----------------------------- */

data class DetectionResult(
    val xCenter: Float, val yCenter: Float,
    val width: Float, val height: Float,
    val confidence: Float
)

data class BoundingBox(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val confidence: Float, val classId: Int
)

/* -----------------------------   global singletons   ----------------------------- */
private var tfliteInterpreter: Interpreter? = null

object Settings {
    object DetectionMode {
        var enableYOLOinference: Boolean = true
    }
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
    }
    object BoundingBox {
        var enableBoundingBox: Boolean = true
        var boxColor: Scalar = Scalar(0.0, 39.0, 76.0)
        var boxThickness: Int = 10
    }

    object ExportData {
        var videoDATA: Boolean = false   // processed‑video recording
        var takePhoto: Boolean = true    // full‑resolution photo
    }
}

/* -----------------------------      Processor       ----------------------------- */

class VideoProcessor(private val context: Context) {

    init {
        initOpenCV()
    }

    /* load OpenCV native lib */
    private fun initOpenCV() {
        try { System.loadLibrary("opencv_java4") }
        catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    /* inject TFLite interpreter */
    fun setInterpreter(model: Interpreter) {
        synchronized(this) { tfliteInterpreter = model }
        Log.d("VideoProcessor", "TFLite model set successfully")
    }

    /* simple reset hint */
    fun reset() = Toast.makeText(context, "VideoProc Reset", Toast.LENGTH_SHORT).show()

    /* entry‑point from MainActivity */
    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try { processFrameInternalYOLO(bitmap) } catch (e: Exception) {
                Log.d("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    /* core work – letterbox, run YOLO, (optionally) draw bounding box */
    private suspend fun processFrameInternalYOLO(src: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {

            val (inputW, inputH, outputShape) = getModelDimensions()
            val (letterboxed, offsets) = YOLOHelper.createLetterboxedBitmap(src, inputW, inputH)

            val dstMat = Mat().also { Utils.bitmapToMat(src, it) }

            if (Settings.DetectionMode.enableYOLOinference && tfliteInterpreter != null) {
                val out = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
                TensorImage(DataType.FLOAT32).apply { load(letterboxed) }
                    .also { tfliteInterpreter?.run(it.buffer, out) }

                YOLOHelper.parseTFLite(out)?.let { det ->
                    val (box, _) = YOLOHelper.rescaleInferencedCoordinates(
                        det, src.width, src.height, offsets, inputW, inputH
                    )
                    if (Settings.BoundingBox.enableBoundingBox) {
                        YOLOHelper.drawBoundingBoxes(dstMat, box)
                    }
                }
            }

            val processedBmp = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
                .also { Utils.matToBitmap(dstMat, it); dstMat.release() }

            processedBmp to letterboxed
        }

    /* helper: get model I/O tensor shapes */
    fun getModelDimensions(): Triple<Int, Int, List<Int>> {
        val inTensor   = tfliteInterpreter?.getInputTensor(0)
        val shapeIn    = inTensor?.shape()
        val inputH     = shapeIn?.getOrNull(1) ?: 416
        val inputW     = shapeIn?.getOrNull(2) ?: 416
        val outTensor  = tfliteInterpreter?.getOutputTensor(0)
        val shapeOut   = outTensor?.shape()?.toList() ?: listOf(1, 5, 3549)
        return Triple(inputW, inputH, shapeOut)
    }
}

/* -----------------------------    Helper Objects    ----------------------------- */

object YOLOHelper {
    /* Parse raw TFLite output, run simple NMS, return best box */
    fun parseTFLite(raw: Array<Array<FloatArray>>): DetectionResult? {
        val n = raw[0][0].size
        val dets = ArrayList<DetectionResult>()
        for (i in 0 until n) {
            val conf = raw[0][4][i]
            if (conf >= Settings.Inference.confidenceThreshold) {
                dets += DetectionResult(
                    raw[0][0][i], raw[0][1][i],
                    raw[0][2][i], raw[0][3][i],
                    conf
                )
            }
        }
        if (dets.isEmpty()) return null

        /* convert to boxes for IoU */
        val boxes = dets.map { it to detectionToBox(it) }.toMutableList()
        boxes.sortByDescending { it.first.confidence }
        val final = mutableListOf<DetectionResult>()
        while (boxes.isNotEmpty()) {
            val best = boxes.removeAt(0)
            final.add(best.first)
            boxes.removeAll { computeIoU(best.second, it.second) > Settings.Inference.iouThreshold }
        }
        return final.maxByOrNull { it.confidence }
    }

    /* convert center‑form to corners */
    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        d.xCenter - d.width / 2, d.yCenter - d.height / 2,
        d.xCenter + d.width / 2, d.yCenter + d.height / 2,
        d.confidence, 0
    )

    /* IoU utility */
    private fun computeIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union > 0f) inter / union else 0f
    }

    /* Transform to original image coords */
    fun rescaleInferencedCoordinates(
        det: DetectionResult,
        origW: Int, origH: Int,
        padOffsets: Pair<Int, Int>,
        inW: Int, inH: Int
    ): Pair<BoundingBox, Point> {
        val scale  = min(inW / origW.toDouble(), inH / origH.toDouble())
        val (padL, padT) = padOffsets
        val cxL = det.xCenter * inW
        val cyL = det.yCenter * inH
        val wL  = det.width  * inW
        val hL  = det.height * inH
        val cxO = (cxL - padL) / scale
        val cyO = (cyL - padT) / scale
        val wO  = wL / scale
        val hO  = hL / scale
        val box = BoundingBox(
            (cxO - wO / 2).toFloat(), (cyO - hO / 2).toFloat(),
            (cxO + wO / 2).toFloat(), (cyO + hO / 2).toFloat(),
            det.confidence, 0
        )
        return box to Point(cxO, cyO)
    }

    /* Draw rectangle & label */
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        Imgproc.rectangle(
            mat,
            Point(box.x1.toDouble(), box.y1.toDouble()),
            Point(box.x2.toDouble(), box.y2.toDouble()),
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )
        val label = "Detected Cube (${("%.1f".format(box.confidence * 100))}%)"
        Imgproc.putText(
            mat, label,
            Point(box.x1.toDouble(), (box.y1 - 5).coerceAtLeast(10f).toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.5,
            Scalar(255.0, 255.0, 255.0), 2
        )
    }

    /* create letter‑boxed square bitmap for YOLO */
    fun createLetterboxedBitmap(
        src: Bitmap,
        targetW: Int, targetH: Int,
        padColor: Scalar = Scalar(0.0, 0.0, 0.0)
    ): Pair<Bitmap, Pair<Int, Int>> {

        val srcM = Mat().also { Utils.bitmapToMat(src, it) }
        val (sW, sH) = srcM.cols().toDouble() to srcM.rows().toDouble()
        val scale = min(targetW / sW, targetH / sH)
        val newW  = (sW * scale).toInt()
        val newH  = (sH * scale).toInt()

        val resized = Mat().also { Imgproc.resize(srcM, it, Size(newW.toDouble(), newH.toDouble())) }
        srcM.release()

        val padW = targetW - newW
        val padH = targetH - newH
        val (top,    bottom) = padH / 2 to (padH - padH / 2)
        val (left,   right)  = padW / 2 to (padW - padW / 2)

        val dstM = Mat().also {
            Core.copyMakeBorder(resized, it, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)
            resized.release()
        }
        val outBmp = Bitmap.createBitmap(dstM.cols(), dstM.rows(), src.config)
            .also { Utils.matToBitmap(dstM, it); dstM.release() }

        return outBmp to (left to top)
    }
}