package com.developer27.ustar.machinelearning.DynaSpa

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object DynaSpaMaskProcessor {

    // OpenCV-only crop extraction controls
    private const val BORDER_RING_WIDTH = 10
    private const val DETAIL_PERCENTILE = 92.5
    private const val SELECTION_CLOSE_KERNEL = 5
    private const val SELECTION_CLOSE_ITERS = 2
    private const val FINAL_GATE_DILATE_KERNEL = 7
    private const val MIN_COMPONENT_AREA = 30

    /**
     * Uses the already-created bounding box:
     * crop -> OpenCV UOID tag extraction -> reconstruct on black background
     */
    fun processBoundingBox(
        source: Bitmap,
        boundingBox: Rect
    ): Bitmap {
        val config = source.config ?: Bitmap.Config.ARGB_8888

        // Black output canvas; only processed bbox content will be restored
        val resultBitmap = Bitmap.createBitmap(source.width, source.height, config)
        resultBitmap.eraseColor(Color.BLACK)

        if (source.width <= 0 || source.height <= 0) {
            return resultBitmap
        }

        // Clamp bbox safely to image bounds
        val left = boundingBox.left.coerceIn(0, source.width - 1)
        val top = boundingBox.top.coerceIn(0, source.height - 1)
        val right = boundingBox.right.coerceIn(left + 1, source.width)
        val bottom = boundingBox.bottom.coerceIn(top + 1, source.height)

        val safeRect = Rect(left, top, right, bottom)

        if (safeRect.width() <= 0 || safeRect.height() <= 0) {
            return resultBitmap
        }

        // Crop only the detected bbox region
        val croppedBitmap = Bitmap.createBitmap(
            source,
            safeRect.left,
            safeRect.top,
            safeRect.width(),
            safeRect.height()
        )

        // Apply OpenCV-only extraction inside the crop
        val processedCrop = applyOpenCvTagExtractionToCrop(croppedBitmap)

        // Put the processed crop back in its original location
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(processedCrop, safeRect.left.toFloat(), safeRect.top.toFloat(), null)

        return resultBitmap
    }

    /**
     * Applies the OpenCV-only tag extraction to the crop.
     * Output is RGB content on black background inside the crop.
     */
    private fun applyOpenCvTagExtractionToCrop(input: Bitmap): Bitmap {
        val rgba = Mat()
        val rgb = Mat()
        val resultRgba = Mat()
        var tagOnlyRgb: Mat? = null

        try {
            Utils.bitmapToMat(input, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

            tagOnlyRgb = extractUoidTagOnlyOpenCv(
                regionRgb = rgb,
                borderRingWidth = BORDER_RING_WIDTH,
                detailPercentile = DETAIL_PERCENTILE,
                selectionCloseKernel = SELECTION_CLOSE_KERNEL,
                selectionCloseIters = SELECTION_CLOSE_ITERS,
                finalGateDilateKernel = FINAL_GATE_DILATE_KERNEL,
                minComponentArea = MIN_COMPONENT_AREA
            )

            Imgproc.cvtColor(tagOnlyRgb, resultRgba, Imgproc.COLOR_RGB2RGBA)

            val outputBitmap = Bitmap.createBitmap(
                resultRgba.cols(),
                resultRgba.rows(),
                Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(resultRgba, outputBitmap)
            return outputBitmap
        } finally {
            tagOnlyRgb?.release()
            resultRgba.release()
            rgba.release()
            rgb.release()
        }
    }

    /**
     * OpenCV-only extraction of the UOID tag from the bbox crop.
     *
     * Strategy:
     * 1) Estimate background color from the border ring in LAB
     * 2) Emphasize thin bright structure using top-hat + CLAHE
     * 3) Add color-distance / edge cues
     * 4) Build a detail mask
     * 5) Build a connected selection mask
     * 6) Keep only the main cube-like component
     * 7) Copy only those pixels into a black RGB output
     */
    private fun extractUoidTagOnlyOpenCv(
        regionRgb: Mat,
        borderRingWidth: Int,
        detailPercentile: Double,
        selectionCloseKernel: Int,
        selectionCloseIters: Int,
        finalGateDilateKernel: Int,
        minComponentArea: Int
    ): Mat {
        val toRelease = mutableListOf<Mat>()
        fun keep(mat: Mat): Mat {
            toRelease.add(mat)
            return mat
        }

        var outputToKeep: Mat? = null

        try {
            val regionBgr = keep(Mat())
            val lab = keep(Mat())
            val hsv = keep(Mat())
            val gray = keep(Mat())
            val grayEq = keep(Mat())
            val grayBlur = keep(Mat())
            val satChannel = keep(Mat())
            val topHat = keep(Mat())
            val canny = keep(Mat())
            val cannyFloat = keep(Mat())
            val lap = keep(Mat())
            val lapAbs = keep(Mat())
            val combined = keep(Mat.zeros(regionRgb.size(), CvType.CV_32F))
            val temp = keep(Mat())
            val detailMaskFloat = keep(Mat())
            val detailMask = keep(Mat())
            val selectionMask = keep(Mat())
            val gate = keep(Mat())
            val finalMask = keep(Mat())

            Imgproc.cvtColor(regionRgb, regionBgr, Imgproc.COLOR_RGB2BGR)
            Imgproc.cvtColor(regionBgr, lab, Imgproc.COLOR_BGR2Lab)
            Imgproc.cvtColor(regionBgr, hsv, Imgproc.COLOR_BGR2HSV)
            Imgproc.cvtColor(regionBgr, gray, Imgproc.COLOR_BGR2GRAY)

            val claheGray = Imgproc.createCLAHE(2.2, Size(8.0, 8.0))
            claheGray.apply(gray, grayEq)

            Imgproc.bilateralFilter(grayEq, grayBlur, 7, 45.0, 45.0)

            val h = gray.rows()
            val w = gray.cols()

            if (h <= 0 || w <= 0) {
                outputToKeep = Mat.zeros(regionRgb.size(), CvType.CV_8UC3)
                return outputToKeep!!
            }

            val ring = max(2, min(borderRingWidth, min(h, w) / 4))

            // Background color estimate from the border ring in LAB
            val bgMedian = computeBorderMedianLab(lab, ring)

            // LAB distance map
            val labDist = keep(createLabDistanceMap(lab, bgMedian))
            val labDistNorm = keep(normalizeMask(labDist))

            // Top-hat on grayscale detail
            val kTopHat = max(11, ensureOdd(min(h, w) / 12))
            val topHatKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(kTopHat.toDouble(), kTopHat.toDouble())
                )
            )
            Imgproc.morphologyEx(grayBlur, topHat, Imgproc.MORPH_TOPHAT, topHatKernel)
            val topHatNorm = keep(normalizeMask(topHat))

            // Edge cues
            Imgproc.Canny(grayBlur, canny, 30.0, 90.0)
            canny.convertTo(cannyFloat, CvType.CV_32F, 1.0 / 255.0)

            Imgproc.Laplacian(grayBlur, lap, CvType.CV_32F, 3)
            Core.absdiff(lap, Scalar.all(0.0), lapAbs)
            val lapNorm = keep(normalizeMask(lapAbs))

            // Saturation cue
            Core.extractChannel(hsv, satChannel, 1)
            val satNorm = keep(normalizeMask(satChannel))

            // Weighted cue fusion:
            // 0.45 * top_hat + 0.25 * lab_dist + 0.15 * canny + 0.10 * lap + 0.05 * sat
            Core.addWeighted(topHatNorm, 0.45, labDistNorm, 0.25, 0.0, combined)
            Core.addWeighted(cannyFloat, 0.15, lapNorm, 0.10, 0.0, temp)
            Core.add(combined, temp, combined)
            Core.multiply(satNorm, Scalar.all(0.05), temp)
            Core.add(combined, temp, combined)

            val combinedNorm = keep(normalizeMask(combined))

            if (Core.minMaxLoc(combinedNorm).maxVal <= 1e-8) {
                outputToKeep = Mat.zeros(regionRgb.size(), CvType.CV_8UC3)
                return outputToKeep!!
            }

            // High-percentile threshold to keep fine detail
            val thrVal = percentile(combinedNorm, detailPercentile)
            Imgproc.threshold(combinedNorm, detailMaskFloat, thrVal, 255.0, Imgproc.THRESH_BINARY)
            detailMaskFloat.convertTo(detailMask, CvType.CV_8U)

            val detailOpenKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(3.0, 3.0)
                )
            )
            Imgproc.morphologyEx(detailMask, detailMask, Imgproc.MORPH_OPEN, detailOpenKernel)

            // Connect nearby detail to find the main region
            val selectionKernelSize = ensureOdd(selectionCloseKernel)
            val selectionKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(selectionKernelSize.toDouble(), selectionKernelSize.toDouble())
                )
            )
            Imgproc.morphologyEx(
                detailMask,
                selectionMask,
                Imgproc.MORPH_CLOSE,
                selectionKernel,
                Point(-1.0, -1.0),
                selectionCloseIters
            )

            val dilate3 = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(3.0, 3.0)
                )
            )
            Imgproc.dilate(selectionMask, selectionMask, dilate3)

            // Keep the single best component, preferring large + central
            val anchorComponent = keep(keepBestComponent(selectionMask, minComponentArea))
            val fallbackAnchor =
                if (Core.countNonZero(anchorComponent) == 0) keep(keepTopConnectedComponents(selectionMask, 1, 1))
                else null

            val gateSource = fallbackAnchor ?: anchorComponent

            val gateKernelSize = ensureOdd(finalGateDilateKernel)
            val gateKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(gateKernelSize.toDouble(), gateKernelSize.toDouble())
                )
            )
            Imgproc.dilate(gateSource, gate, gateKernel)

            // Keep only detailed pixels that belong to the main gated region
            Core.bitwise_and(detailMask, gate, finalMask)

            val finalCloseKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(3.0, 3.0)
                )
            )
            Imgproc.morphologyEx(finalMask, finalMask, Imgproc.MORPH_CLOSE, finalCloseKernel)

            val cleanedFinal = keep(
                keepTopConnectedComponents(
                    finalMask,
                    4,
                    max(8, minComponentArea / 3)
                )
            )

            // Ensure clean binary mask
            Imgproc.threshold(cleanedFinal, cleanedFinal, 0.0, 255.0, Imgproc.THRESH_BINARY)

            // Copy only selected pixels onto black background
            outputToKeep = Mat.zeros(regionRgb.size(), CvType.CV_8UC3)
            regionRgb.copyTo(outputToKeep, cleanedFinal)

            return outputToKeep!!
        } finally {
            toRelease.forEach { mat ->
                if (mat !== outputToKeep) {
                    mat.release()
                }
            }
        }
    }

    /**
     * Compute the median LAB color from the crop border ring.
     */
    private fun computeBorderMedianLab(lab: Mat, ring: Int): DoubleArray {
        val lValues = mutableListOf<Double>()
        val aValues = mutableListOf<Double>()
        val bValues = mutableListOf<Double>()

        val h = lab.rows()
        val w = lab.cols()

        for (y in 0 until h) {
            for (x in 0 until w) {
                if (y < ring || y >= h - ring || x < ring || x >= w - ring) {
                    val px = lab.get(y, x) ?: continue
                    if (px.size >= 3) {
                        lValues.add(px[0])
                        aValues.add(px[1])
                        bValues.add(px[2])
                    }
                }
            }
        }

        return doubleArrayOf(
            median(lValues),
            median(aValues),
            median(bValues)
        )
    }

    /**
     * Build a single-channel LAB distance map:
     * distance(pixelLAB, borderMedianLAB)
     */
    private fun createLabDistanceMap(lab: Mat, bgMedian: DoubleArray): Mat {
        val h = lab.rows()
        val w = lab.cols()
        val output = Mat.zeros(lab.size(), CvType.CV_32F)

        for (y in 0 until h) {
            val rowBuffer = FloatArray(w)
            for (x in 0 until w) {
                val px = lab.get(y, x) ?: doubleArrayOf(0.0, 0.0, 0.0)

                val dl = px[0] - bgMedian[0]
                val da = px[1] - bgMedian[1]
                val db = px[2] - bgMedian[2]

                rowBuffer[x] = sqrt(dl * dl + da * da + db * db).toFloat()
            }
            output.put(y, 0, rowBuffer)
        }

        return output
    }

    /**
     * Normalize a single-channel Mat to [0,1].
     */
    private fun normalizeMask(src: Mat): Mat {
        val srcFloat = Mat()
        val normalized = Mat()

        try {
            src.convertTo(srcFloat, CvType.CV_32F)

            val mm = Core.minMaxLoc(srcFloat)
            val minVal = mm.minVal
            val maxVal = mm.maxVal
            val range = maxVal - minVal

            if (range < 1e-8) {
                normalized.create(src.size(), CvType.CV_32F)
                normalized.setTo(Scalar.all(0.0))
            } else {
                Core.subtract(srcFloat, Scalar.all(minVal), normalized)
                Core.multiply(normalized, Scalar.all(1.0 / range), normalized)
            }

            return normalized
        } finally {
            srcFloat.release()
        }
    }

    /**
     * Keep the largest connected components from a binary 8U mask.
     */
    private fun keepTopConnectedComponents(
        binaryMask: Mat,
        maxComponents: Int = 3,
        minArea: Int = 10
    ): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        try {
            val numLabels = Imgproc.connectedComponentsWithStats(
                binaryMask,
                labels,
                stats,
                centroids,
                8,
                CvType.CV_32S
            )

            if (numLabels <= 1) {
                return binaryMask.clone()
            }

            val ranked = mutableListOf<Pair<Int, Int>>()

            for (i in 1 until numLabels) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
                if (area >= minArea) {
                    ranked.add(i to area)
                }
            }

            if (ranked.isEmpty()) {
                return Mat.zeros(binaryMask.size(), CvType.CV_8U)
            }

            ranked.sortByDescending { it.second }
            val keepIndices = ranked.take(maxComponents).map { it.first }.toSet()

            return buildMaskFromLabels(labels, keepIndices, binaryMask.size())
        } finally {
            labels.release()
            stats.release()
            centroids.release()
        }
    }

    /**
     * Keep the single best component, preferring a large component near the crop center.
     */
    private fun keepBestComponent(
        binaryMask: Mat,
        minArea: Int = 30
    ): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        try {
            val numLabels = Imgproc.connectedComponentsWithStats(
                binaryMask,
                labels,
                stats,
                centroids,
                8,
                CvType.CV_32S
            )

            if (numLabels <= 1) {
                return Mat.zeros(binaryMask.size(), CvType.CV_8U)
            }

            val centerX = binaryMask.cols() / 2.0
            val centerY = binaryMask.rows() / 2.0
            val diag = hypot(binaryMask.cols().toDouble(), binaryMask.rows().toDouble()) + 1e-8

            var bestIdx = -1
            var bestScore = Double.NEGATIVE_INFINITY

            for (i in 1 until numLabels) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                if (area < minArea) {
                    continue
                }

                val cx = centroids.get(i, 0)[0]
                val cy = centroids.get(i, 1)[0]

                val dist = hypot(cx - centerX, cy - centerY) / diag
                val score = area * (1.35 - dist)

                if (score > bestScore) {
                    bestScore = score
                    bestIdx = i
                }
            }

            if (bestIdx < 0) {
                return Mat.zeros(binaryMask.size(), CvType.CV_8U)
            }

            return buildMaskFromLabels(labels, setOf(bestIdx), binaryMask.size())
        } finally {
            labels.release()
            stats.release()
            centroids.release()
        }
    }

    /**
     * Build an 8U mask from selected component labels.
     */
    private fun buildMaskFromLabels(
        labels: Mat,
        keepIndices: Set<Int>,
        outputSize: Size
    ): Mat {
        val resultMask = Mat.zeros(outputSize, CvType.CV_8U)

        if (keepIndices.isEmpty()) {
            return resultMask
        }

        val labelData = IntArray(labels.rows() * labels.cols())
        labels.get(0, 0, labelData)

        val maskData = ByteArray(labelData.size)
        for (i in labelData.indices) {
            if (labelData[i] in keepIndices) {
                maskData[i] = 255.toByte()
            }
        }

        resultMask.put(0, 0, maskData)
        return resultMask
    }

    /**
     * Percentile for a single-channel Mat.
     */
    private fun percentile(src: Mat, percentile: Double): Double {
        val srcFloat = Mat()

        try {
            src.convertTo(srcFloat, CvType.CV_32F)

            val total = srcFloat.rows() * srcFloat.cols()
            if (total <= 0) {
                return 0.0
            }

            val data = FloatArray(total)

            if (srcFloat.isContinuous) {
                srcFloat.get(0, 0, data)
            } else {
                var offset = 0
                val rowBuffer = FloatArray(srcFloat.cols())
                for (r in 0 until srcFloat.rows()) {
                    srcFloat.get(r, 0, rowBuffer)
                    System.arraycopy(rowBuffer, 0, data, offset, rowBuffer.size)
                    offset += rowBuffer.size
                }
            }

            data.sort()

            val p = percentile.coerceIn(0.0, 100.0)
            val position = (p / 100.0) * (data.size - 1)
            val lo = floor(position).toInt()
            val hi = ceil(position).toInt()

            if (lo == hi) {
                return data[lo].toDouble()
            }

            val weight = position - lo
            return data[lo] * (1.0 - weight) + data[hi] * weight
        } finally {
            srcFloat.release()
        }
    }

    /**
     * Median of a list of doubles.
     */
    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) {
            return 0.0
        }

        val sorted = values.sorted()
        val mid = sorted.size / 2

        return if (sorted.size % 2 == 0) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /**
     * Force an odd integer >= 3.
     */
    private fun ensureOdd(v: Int): Int {
        val clamped = max(3, v)
        return if (clamped % 2 == 1) clamped else clamped + 1
    }
}