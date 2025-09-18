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

/**
 * Utility helpers around YOLO-style TFLite outputs:
 *  - Parse raw TFLite output tensors into simple detections.
 *  - Apply a tiny Non‑Maximum Suppression (NMS).
 *  - Map model-space detections to a target image space.
 *  - Draw OpenCV rectangles + labels.
 *
 * IMPORTANT COORDINATE NOTE
 * -------------------------
 * This helper assumes the model outputs CENTER-FORM boxes: (x_center, y_center, width, height, confidence).
 * It also assumes the outputs are *normalized* in [0,1] unless stated otherwise.
 *
 * Image Size Assumptions
 * ----------------------
 * The pipeline in VideoProcessor stretches the input frame to 640×640 (no letterboxing).
 * That means aspect ratio is not preserved. If your model was trained with letterbox padding,
 * you might want to mirror that at inference time and adjust the mapping accordingly.
 */
object YOLOHelper {

    /**
     * Parse raw TFLite output into a list of candidate detections and run a lightweight NMS.
     *
     * Supported output shapes (rank-3):
     *   - [1, 5, N]: for each i in N → [x, y, w, h, conf] laid out along the second dim.
     *   - [1, N, 5]: for each i in N → [x, y, w, h, conf] laid out along the third dim.
     *
     * Returns:
     *   The *single* best DetectionResult after NMS (highest confidence among kept), or null if none.
     *
     * If you want multiple boxes:
     *   - Replace the final line with `return kept` and update the call site accordingly.
     */
    fun parseTFLite(raw: Array<Array<FloatArray>>): DetectionResult? {
        if (raw.isEmpty() || raw[0].isEmpty()) return null

        val dim1 = raw[0].size          // either 5 or N
        val dim2 = raw[0][0].size       // either N or 5

        val dets = ArrayList<DetectionResult>()
        when {
            // Case A: [1, 5, N] tensor layout
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
            // Case B: [1, N, 5] tensor layout
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
            else -> return null // unknown layout
        }
        if (dets.isEmpty()) return null

        // ---- NMS (on corner-form boxes derived from center-form) ----
        // 1) Convert to (x1,y1,x2,y2) to make IoU computation straightforward.
        // 2) Sort by confidence descending.
        // 3) Keep the best, discard any box with IoU > threshold wrt a kept box.
        val boxes = dets.map { it to detectionToBox(it) }.toMutableList()
        boxes.sortByDescending { it.first.confidence }

        val kept = mutableListOf<DetectionResult>()
        while (boxes.isNotEmpty()) {
            val best = boxes.removeAt(0) // highest remaining confidence
            kept += best.first
            // Remove boxes that overlap too much with the "best" one
            boxes.removeAll { computeIoU(best.second, it.second) > Settings.Inference.iouThreshold }
        }

        // This helper returns the single best detection.
        // If you want all kept, return them instead.
        return kept.maxByOrNull { it.confidence }
    }

    /** Convert a center-form detection into a corner-form bounding box. */
    private fun detectionToBox(d: DetectionResult) = BoundingBox(
        x1 = d.xCenter - d.width / 2f,
        y1 = d.yCenter - d.height / 2f,
        x2 = d.xCenter + d.width / 2f,
        y2 = d.yCenter + d.height / 2f,
        confidence = d.confidence,
        classId = 0 // single-class usage; adjust if your model emits multiple classes
    )

    /**
     * Compute Intersection-over-Union (IoU) for two corner-form boxes.
     * Returns 0 if there is no overlap or union area is zero.
     */
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

    /**
     * Preprocess an input Bitmap for the model.
     *
     * Current behavior:
     *   1) Optionally fix EXIF orientation (placeholder here).
     *   2) Stretch image to 640×640 (aspect ratio is NOT preserved).
     *
     * If your model expects a specific normalization (e.g., divide by 255, mean/std),
     * do that at the TensorImage creation step in VideoProcessor (or here).
     */
    fun preprocessInput(src: Bitmap): Bitmap {
        // 1) Auto-orient if needed (hook EXIF here if loading from file)
        val upright = src // replace with EXIF-based rotation if your inputs come from disk

        // 2) Stretch to 640×640 (no letterbox). Change if you need letterboxing.
        return Bitmap.createScaledBitmap(upright, 640, 640, true)
    }

    /**
     * Map a detection from model coordinates to the target image space (e.g., 640×640 overlay).
     *
     * @param det                 Detection in center-form.
     * @param targetW,targetH     Size of the target canvas (where you will draw).
     * @param inputW,inputH       Model input tensor size (used when boxesAreNormalized=false).
     * @param boxesAreNormalized  true if (x,y,w,h) are in [0,1] relative to model input.
     *
     * Returns: RectF in target image coordinates, clamped to the target bounds.
     */
    fun toTargetRect(
        det: DetectionResult,
        targetW: Int,
        targetH: Int,
        inputW: Int,
        inputH: Int,
        boxesAreNormalized: Boolean = true
    ): RectF {
        // Scale center, width, height from model space to target space.
        // If outputs are normalized, scale by target size directly.
        // Otherwise scale by the ratio between target and the model input tensor.
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

        // Convert back to corner-form and clamp to the canvas.
        val left   = (cx - w / 2f).coerceIn(0f, targetW.toFloat())
        val top    = (cy - h / 2f).coerceIn(0f, targetH.toFloat())
        val right  = (cx + w / 2f).coerceIn(0f, targetW.toFloat())
        val bottom = (cy + h / 2f).coerceIn(0f, targetH.toFloat())

        return RectF(left, top, right, bottom)
    }

    /**
     * Draw one rectangle + confidence label onto an OpenCV Mat (BGR color space).
     *
     * @param mat          The OpenCV Mat backing your overlay bitmap (expects 640×640 in this pipeline).
     * @param rect         RectF in the *original image space* that this Mat represents.
     * @param confidence   [0,1] confidence for the label string.
     * @param labelPrefix  Optional class label (default "3D Cube").
     */
    fun drawBoundingBoxes(mat: Mat, rect: RectF, confidence: Float, labelPrefix: String = "3D Cube") {
        // OpenCV Scalar is BGR (not RGB).
        Imgproc.rectangle(
            mat,
            Point(rect.left.toDouble(),  rect.top.toDouble()),
            Point(rect.right.toDouble(), rect.bottom.toDouble()),
            Settings.BoundingBox.boxColor,
            Settings.BoundingBox.boxThickness
        )

        // Simple "Label (xx.x%)" text, placed just above the top-left corner if possible.
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