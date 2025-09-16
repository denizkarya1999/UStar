package com.developer27.ustar.videoprocessing

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

object YOLOHelper {

    /** Parse raw TFLite output into detections and run a light NMS.
     *  Supports [1,5,N] and [1,N,5] (x,y,w,h,conf). */
    fun parseTFLite(raw: Array<Array<FloatArray>>): DetectionResult? {
        if (raw.isEmpty() || raw[0].isEmpty()) return null

        val dim1 = raw[0].size
        val dim2 = raw[0][0].size

        val dets = ArrayList<DetectionResult>()
        when {
            // [1, 5, N]
            dim1 == 5 -> {
                val n = dim2
                for (i in 0 until n) {
                    val x = raw[0][0][i]
                    val y = raw[0][1][i]
                    val w = raw[0][2][i]
                    val h = raw[0][3][i]
                    val conf = raw[0][4][i]
                    if (conf >= Settings.Inference.confidenceThreshold) {
                        dets += DetectionResult(x, y, w, h, conf)
                    }
                }
            }
            // [1, N, 5]
            dim2 == 5 -> {
                val n = dim1
                for (i in 0 until n) {
                    val x = raw[0][i][0]
                    val y = raw[0][i][1]
                    val w = raw[0][i][2]
                    val h = raw[0][i][3]
                    val conf = raw[0][i][4]
                    if (conf >= Settings.Inference.confidenceThreshold) {
                        dets += DetectionResult(x, y, w, h, conf)
                    }
                }
            }
            else -> return null
        }
        if (dets.isEmpty()) return null

        // NMS over corner-form boxes derived from center-form
        val boxes = dets.map { it to detectionToBox(it) }.toMutableList()
        boxes.sortByDescending { it.first.confidence }

        val kept = mutableListOf<DetectionResult>()
        while (boxes.isNotEmpty()) {
            val best = boxes.removeAt(0)
            kept += best.first
            boxes.removeAll { computeIoU(best.second, it.second) > Settings.Inference.iouThreshold }
        }
        return kept.maxByOrNull { it.confidence }
    }

    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2f,
        y1 = d.yCenter - d.height / 2f,
        x2 = d.xCenter + d.width / 2f,
        y2 = d.yCenter + d.height / 2f,
        confidence = d.confidence,
        classId = 0
    )

    private fun computeIoU(a: BoundingBox, b: BoundingBox): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val interW = max(0f, x2 - x1)
        val interH = max(0f, y2 - y1)
        val inter = interW * interH
        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }

    fun preprocessInput(src: Bitmap): Bitmap {
        // 1) Auto-orient if needed (hook Exif here if loading from file)
        val upright = src // replace with EXIF-based rotation if your inputs come from disk

        // 2) Stretch to 640×640
        return Bitmap.createScaledBitmap(upright, 640, 640, true)
    }

    fun toTargetRect(
        det: DetectionResult,
        targetW: Int,
        targetH: Int,
        inputW: Int,
        inputH: Int,
        boxesAreNormalized: Boolean = true
    ): android.graphics.RectF {
        // Scale center, width, height depending on whether model outputs are normalized
        val cx = if (boxesAreNormalized) {
            det.xCenter * targetW
        } else {
            det.xCenter * (targetW.toFloat() / inputW.toFloat())
        }
        val cy = if (boxesAreNormalized) {
            det.yCenter * targetH
        } else {
            det.yCenter * (targetH.toFloat() / inputH.toFloat())
        }
        val w = if (boxesAreNormalized) {
            det.width * targetW
        } else {
            det.width * (targetW.toFloat() / inputW.toFloat())
        }
        val h = if (boxesAreNormalized) {
            det.height * targetH
        } else {
            det.height * (targetH.toFloat() / inputH.toFloat())
        }

        val left   = (cx - w / 2f).coerceIn(0f, targetW.toFloat())
        val top    = (cy - h / 2f).coerceIn(0f, targetH.toFloat())
        val right  = (cx + w / 2f).coerceIn(0f, targetW.toFloat())
        val bottom = (cy + h / 2f).coerceIn(0f, targetH.toFloat())

        return RectF(left, top, right, bottom)
    }

    /** Draw a rectangle (RectF in original image space) + label with confidence. */
    fun drawBoundingBoxes(mat: Mat, rect: RectF, confidence: Float, labelPrefix: String = "3D Cube") {
        Imgproc.rectangle(
            mat,
            Point(rect.left.toDouble(),  rect.top.toDouble()),
            Point(rect.right.toDouble(), rect.bottom.toDouble()),
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )
        val label = "$labelPrefix (${("%.1f".format(confidence * 100f))}%)"
        val textOrg = Point(
            rect.left.toDouble(),
            max(10f, rect.top - 5f).toDouble()
        )
        Imgproc.putText(
            mat, label, textOrg,
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.5,
            Scalar(255.0, 255.0, 255.0), 2
        )
    }
}