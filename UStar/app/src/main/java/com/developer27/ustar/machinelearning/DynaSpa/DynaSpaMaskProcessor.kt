package com.developer27.ustar.machinelearning.DynaSpa

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object DynaSpaMaskProcessor {

    private const val TAG = "DynaSpaMaskProcessor"

    @Volatile
    private var openCvInitialized = false

    // Continuous OpenCV crop extraction controls
    private const val BORDER_RING_WIDTH = 12
    private const val WEAK_PERCENTILE = 79.0
    private const val STRONG_PERCENTILE = 91.0
    private const val SUPPORT_CLOSE_KERNEL = 7
    private const val SUPPORT_CLOSE_ITERS = 2
    private const val SUPPORT_DILATE_KERNEL = 5
    private const val MIN_COMPONENT_AREA = 30
    private const val FILL_HOLE_MAX_AREA = 140
    private const val GRABCUT_ITERS = 4
    private const val DENOISE_H = 5.0
    private const val ALPHA_FLOOR = 0.12
    private const val ALPHA_SPAN = 0.62
    private const val ALPHA_BLUR_SIGMA = 1.2
    private const val SEED_PERCENTILE_INSIDE_SUPPORT = 55.0
    private const val INPAINT_RADIUS = 3.0

    /**
     * Make sure OpenCV native library is loaded before any Mat() call.
     */
    @Synchronized
    private fun ensureOpenCvLoaded() {
        if (openCvInitialized) return

        var loaded = false

        try {
            loaded = OpenCVLoader.initDebug()
            Log.d(TAG, "OpenCV initDebug(): $loaded")
        } catch (_: Throwable) {
        }

        if (!loaded) {
            try {
                System.loadLibrary("opencv_java4")
                loaded = true
                Log.d(TAG, "OpenCV loaded with System.loadLibrary(opencv_java4)")
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load OpenCV with opencv_java4", t)
            }
        }

        if (!loaded) {
            throw UnsatisfiedLinkError(
                "OpenCV native library could not be loaded. " +
                        "Initialize OpenCV at app startup or bundle the correct native OpenCV library."
            )
        }

        openCvInitialized = true
    }

    /**
     * Uses the already-created bounding box:
     * crop -> continuous OpenCV UOID tag reconstruction -> reconstruct on black background
     */
    fun processBoundingBox(
        source: Bitmap,
        boundingBox: Rect
    ): Bitmap {
        ensureOpenCvLoaded()

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

        // Apply continuous OpenCV reconstruction inside the crop
        val processedCrop = applyOpenCvTagExtractionToCrop(croppedBitmap)

        // Put the processed crop back in its original location
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(processedCrop, safeRect.left.toFloat(), safeRect.top.toFloat(), null)

        return resultBitmap
    }

    /**
     * Applies the continuous OpenCV tag reconstruction to the crop.
     * Output is RGB content on black background inside the crop.
     */
    private fun applyOpenCvTagExtractionToCrop(input: Bitmap): Bitmap {
        ensureOpenCvLoaded()

        val rgba = Mat()
        val rgb = Mat()
        val resultRgba = Mat()
        var tagOnlyRgb: Mat? = null

        try {
            Utils.bitmapToMat(input, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

            tagOnlyRgb = extractUoidTagOnlyOpenCvContinuous(
                regionRgb = rgb,
                borderRingWidth = BORDER_RING_WIDTH,
                weakPercentile = WEAK_PERCENTILE,
                strongPercentile = STRONG_PERCENTILE,
                supportCloseKernel = SUPPORT_CLOSE_KERNEL,
                supportCloseIters = SUPPORT_CLOSE_ITERS,
                supportDilateKernel = SUPPORT_DILATE_KERNEL,
                minComponentArea = MIN_COMPONENT_AREA,
                fillHoleMaxArea = FILL_HOLE_MAX_AREA,
                grabcutIters = GRABCUT_ITERS,
                denoiseH = DENOISE_H,
                alphaFloor = ALPHA_FLOOR,
                alphaSpan = ALPHA_SPAN,
                alphaBlurSigma = ALPHA_BLUR_SIGMA,
                seedPercentileInsideSupport = SEED_PERCENTILE_INSIDE_SUPPORT,
                inpaintRadius = INPAINT_RADIUS
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
     * Continuous OpenCV-only extraction / reconstruction of the UOID tag from the bbox crop.
     *
     * Strategy:
     * 1) Light denoising
     * 2) Estimate background color from the border ring in LAB
     * 3) Build a continuous support region from multiple cues
     * 4) Refine with GrabCut
     * 5) Fill small holes
     * 6) Build soft alpha
     * 7) Reconstruct missing areas with inpainting
     * 8) Copy only reconstructed tag pixels into black RGB output
     */
    private fun extractUoidTagOnlyOpenCvContinuous(
        regionRgb: Mat,
        borderRingWidth: Int,
        weakPercentile: Double,
        strongPercentile: Double,
        supportCloseKernel: Int,
        supportCloseIters: Int,
        supportDilateKernel: Int,
        minComponentArea: Int,
        fillHoleMaxArea: Int,
        grabcutIters: Int,
        denoiseH: Double,
        alphaFloor: Double,
        alphaSpan: Double,
        alphaBlurSigma: Double,
        seedPercentileInsideSupport: Double,
        inpaintRadius: Double
    ): Mat {
        val toRelease = mutableListOf<Mat>()
        fun keep(mat: Mat): Mat {
            toRelease.add(mat)
            return mat
        }

        var outputToKeep: Mat? = null

        try {
            val regionBgr = keep(Mat())
            val denoisedBgr = keep(Mat())
            val lab = keep(Mat())
            val hsv = keep(Mat())
            val gray = keep(Mat())
            val grayEq = keep(Mat())
            val grayBlur = keep(Mat())

            val opened = keep(Mat())
            val whiteBoost = keep(Mat())
            val satChannel = keep(Mat())
            val lap = keep(Mat())
            val lapAbs = keep(Mat())

            val score = keep(Mat.zeros(regionRgb.size(), CvType.CV_32F))
            val tmp = keep(Mat())
            val tmp2 = keep(Mat())

            val weakMaskFloat = keep(Mat())
            val weakMask = keep(Mat())
            val strongMaskFloat = keep(Mat())
            val strongMask = keep(Mat())

            val supportMask = keep(Mat())
            val supportMaskFilled = keep(Mat())
            val alpha = keep(Mat())
            val alphaBlurred = keep(Mat())
            val alphaU8 = keep(Mat())
            val hardMask = keep(Mat())

            val seedMaskFloat = keep(Mat())
            val seedMask = keep(Mat())
            val sparseSeed = keep(Mat.zeros(regionRgb.size(), CvType.CV_8UC3))
            val holeMask = keep(Mat())
            val reconstructedBgr = keep(Mat())
            val reconstructedRgb = keep(Mat())
            val reconstructedFloat = keep(Mat())
            val alphaFloat3 = keep(Mat())
            val finalRgbFloat = keep(Mat())
            val finalRgb = keep(Mat())

            Imgproc.cvtColor(regionRgb, regionBgr, Imgproc.COLOR_RGB2BGR)

            Photo.fastNlMeansDenoisingColored(
                regionBgr,
                denoisedBgr,
                denoiseH.toFloat(),
                denoiseH.toFloat(),
                7,
                21
            )

            Imgproc.cvtColor(denoisedBgr, lab, Imgproc.COLOR_BGR2Lab)
            Imgproc.cvtColor(denoisedBgr, hsv, Imgproc.COLOR_BGR2HSV)
            Imgproc.cvtColor(denoisedBgr, gray, Imgproc.COLOR_BGR2GRAY)

            val claheGray = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            claheGray.apply(gray, grayEq)

            Imgproc.bilateralFilter(grayEq, grayBlur, 7, 35.0, 35.0)

            val h = gray.rows()
            val w = gray.cols()

            if (h <= 0 || w <= 0) {
                outputToKeep = Mat.zeros(regionRgb.size(), CvType.CV_8UC3)
                return outputToKeep!!
            }

            val ring = max(2, min(borderRingWidth, min(h, w) / 4))

            // Background color estimate from the border ring in LAB
            val bgMedian = computeBorderMedianLab(lab, ring)

            // Cue 1: LAB distance from border background
            val labDist = keep(createLabDistanceMap(lab, bgMedian))
            val labDistNorm = keep(normalizeMask(labDist))

            // Cue 2: white / bright structure emphasis
            val kOpen = max(11, ensureOdd(min(h, w) / 5))
            val openKernel = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(kOpen.toDouble(), kOpen.toDouble())
                )
            )
            Imgproc.morphologyEx(grayBlur, opened, Imgproc.MORPH_OPEN, openKernel)
            Core.subtract(grayBlur, opened, whiteBoost)
            whiteBoost.convertTo(whiteBoost, CvType.CV_32F)
            val whiteBoostNorm = keep(normalizeMask(whiteBoost))

            // Cue 3: saturation
            Core.extractChannel(hsv, satChannel, 1)
            val satNorm = keep(normalizeMask(satChannel))

            // Cue 4: structural edges
            Imgproc.Laplacian(grayBlur, lap, CvType.CV_32F, 3)
            Core.absdiff(lap, Scalar.all(0.0), lapAbs)
            val lapNorm = keep(normalizeMask(lapAbs))

            // score = 0.36 * lab_dist + 0.28 * white_boost + 0.22 * sat + 0.14 * grad
            Core.addWeighted(labDistNorm, 0.36, whiteBoostNorm, 0.28, 0.0, score)
            Core.addWeighted(satNorm, 0.22, lapNorm, 0.14, 0.0, tmp)
            Core.add(score, tmp, score)

            val scoreNorm = keep(normalizeMask(score))

            if (Core.minMaxLoc(scoreNorm).maxVal <= 1e-8) {
                outputToKeep = Mat.zeros(regionRgb.size(), CvType.CV_8UC3)
                return outputToKeep!!
            }

            // Weak mask
            val weakThr = percentile(scoreNorm, weakPercentile)
            Imgproc.threshold(scoreNorm, weakMaskFloat, weakThr, 255.0, Imgproc.THRESH_BINARY)
            weakMaskFloat.convertTo(weakMask, CvType.CV_8U)

            val supportCloseKernelMat = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(
                        ensureOdd(supportCloseKernel).toDouble(),
                        ensureOdd(supportCloseKernel).toDouble()
                    )
                )
            )
            Imgproc.morphologyEx(
                weakMask,
                weakMask,
                Imgproc.MORPH_CLOSE,
                supportCloseKernelMat,
                Point(-1.0, -1.0),
                supportCloseIters
            )

            val supportDilateKernelMat = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(
                        ensureOdd(supportDilateKernel).toDouble(),
                        ensureOdd(supportDilateKernel).toDouble()
                    )
                )
            )
            Imgproc.dilate(weakMask, weakMask, supportDilateKernelMat)

            val bestWeak = keep(keepBestComponent(weakMask, minComponentArea))
            if (Core.countNonZero(bestWeak) > 0) {
                bestWeak.copyTo(weakMask)
            } else {
                val fallbackWeakThr = percentile(scoreNorm, 75.0)
                Imgproc.threshold(scoreNorm, weakMaskFloat, fallbackWeakThr, 255.0, Imgproc.THRESH_BINARY)
                weakMaskFloat.convertTo(weakMask, CvType.CV_8U)
                val fallbackWeak = keep(keepTopConnectedComponents(weakMask, 1, 8))
                fallbackWeak.copyTo(weakMask)
            }

            // Strong mask
            val strongThr = percentile(scoreNorm, strongPercentile)
            Imgproc.threshold(scoreNorm, strongMaskFloat, strongThr, 255.0, Imgproc.THRESH_BINARY)
            strongMaskFloat.convertTo(strongMask, CvType.CV_8U)
            Core.bitwise_and(strongMask, weakMask, strongMask)

            val open3 = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(3.0, 3.0)
                )
            )
            Imgproc.morphologyEx(strongMask, strongMask, Imgproc.MORPH_OPEN, open3)
            Imgproc.dilate(strongMask, strongMask, open3)

            if (Core.countNonZero(strongMask) == 0) {
                Imgproc.erode(weakMask, strongMask, open3)
            }

            // GrabCut refinement
            val gcMask = keep(Mat(regionRgb.size(), CvType.CV_8U, Scalar.all(Imgproc.GC_PR_BGD.toDouble())))

            // border = definite background
            for (y in 0 until h) {
                for (x in 0 until w) {
                    if (y < ring || y >= h - ring || x < ring || x >= w - ring) {
                        gcMask.put(y, x, Imgproc.GC_BGD.toDouble())
                    }
                }
            }

            // weak = probable foreground, strong = definite foreground
            for (y in 0 until h) {
                for (x in 0 until w) {
                    if ((weakMask.get(y, x)?.get(0)?.toInt() ?: 0) > 0) {
                        gcMask.put(y, x, Imgproc.GC_PR_FGD.toDouble())
                    }
                    if ((strongMask.get(y, x)?.get(0)?.toInt() ?: 0) > 0) {
                        gcMask.put(y, x, Imgproc.GC_FGD.toDouble())
                    }
                }
            }

            val bgdModel = keep(Mat.zeros(1, 65, CvType.CV_64F))
            val fgdModel = keep(Mat.zeros(1, 65, CvType.CV_64F))

            val grabcutMask = keep(Mat.zeros(regionRgb.size(), CvType.CV_8U))
            try {
                Imgproc.grabCut(
                    denoisedBgr,
                    gcMask,
                    org.opencv.core.Rect(),
                    bgdModel,
                    fgdModel,
                    max(1, grabcutIters),
                    Imgproc.GC_INIT_WITH_MASK
                )

                for (y in 0 until h) {
                    for (x in 0 until w) {
                        val v = gcMask.get(y, x)?.get(0)?.toInt() ?: Imgproc.GC_BGD
                        val isFg = (v == Imgproc.GC_FGD || v == Imgproc.GC_PR_FGD)
                        grabcutMask.put(y, x, if (isFg) 255.0 else 0.0)
                    }
                }
            } catch (_: Exception) {
                weakMask.copyTo(grabcutMask)
            }

            Core.bitwise_or(weakMask, grabcutMask, supportMask)

            val close7 = keep(
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(7.0, 7.0)
                )
            )
            Imgproc.morphologyEx(supportMask, supportMask, Imgproc.MORPH_CLOSE, close7, Point(-1.0, -1.0), 2)

            val filledSupport = keep(fillSmallHoles(supportMask, fillHoleMaxArea))
            val bestSupport = keep(keepBestComponent(filledSupport, minComponentArea))

            if (Core.countNonZero(bestSupport) > 0) {
                bestSupport.copyTo(supportMaskFilled)
            } else {
                weakMask.copyTo(supportMaskFilled)
            }

            // Soft alpha
            scoreNorm.convertTo(alpha, CvType.CV_32F)
            Core.subtract(alpha, Scalar.all(alphaFloor), alpha)
            Core.multiply(alpha, Scalar.all(1.0 / max(1e-6, alphaSpan)), alpha)

            clampMat(alpha, 0.0, 1.0)

            val supportFloat = keep(Mat())
            supportMaskFilled.convertTo(supportFloat, CvType.CV_32F, 1.0 / 255.0)
            Core.multiply(alpha, supportFloat, alpha)

            if (alphaBlurSigma > 0.0) {
                Imgproc.GaussianBlur(alpha, alphaBlurred, Size(0.0, 0.0), alphaBlurSigma)
            } else {
                alpha.copyTo(alphaBlurred)
            }

            clampMat(alphaBlurred, 0.0, 1.0)
            alphaBlurred.convertTo(alphaU8, CvType.CV_8U, 255.0)

            Imgproc.threshold(alphaU8, hardMask, 18.0, 255.0, Imgproc.THRESH_BINARY)

            // Seed mask for inpainting
            val seedThr = if (Core.countNonZero(supportMaskFilled) > 0) {
                percentileInsideMask(scoreNorm, supportMaskFilled, seedPercentileInsideSupport)
            } else {
                percentile(scoreNorm, seedPercentileInsideSupport)
            }

            Imgproc.threshold(scoreNorm, seedMaskFloat, seedThr, 255.0, Imgproc.THRESH_BINARY)
            seedMaskFloat.convertTo(seedMask, CvType.CV_8U)
            Core.bitwise_and(seedMask, supportMaskFilled, seedMask)
            Imgproc.morphologyEx(seedMask, seedMask, Imgproc.MORPH_CLOSE, open3)

            denoisedBgr.copyTo(sparseSeed, seedMask)

            val invSeed = keep(Mat())
            Core.bitwise_not(seedMask, invSeed)
            Core.bitwise_and(supportMaskFilled, invSeed, holeMask)

            Photo.inpaint(sparseSeed, holeMask, reconstructedBgr, inpaintRadius, Photo.INPAINT_TELEA)
            Core.addWeighted(reconstructedBgr, 0.68, denoisedBgr, 0.32, 0.0, reconstructedBgr)

            val invSupport = keep(Mat())
            Core.bitwise_not(supportMaskFilled, invSupport)
            reconstructedBgr.setTo(Scalar.all(0.0), invSupport)

            Imgproc.cvtColor(reconstructedBgr, reconstructedRgb, Imgproc.COLOR_BGR2RGB)

            reconstructedRgb.convertTo(reconstructedFloat, CvType.CV_32FC3)

            val alphaChannels = ArrayList<Mat>(3)
            alphaBlurred.convertTo(tmp2, CvType.CV_32F)
            alphaChannels.add(tmp2.clone())
            alphaChannels.add(tmp2.clone())
            alphaChannels.add(tmp2.clone())
            Core.merge(alphaChannels, alphaFloat3)
            alphaChannels.forEach { it.release() }

            Core.multiply(reconstructedFloat, alphaFloat3, finalRgbFloat)
            finalRgbFloat.convertTo(finalRgb, CvType.CV_8UC3)

            outputToKeep = finalRgb.clone()
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
     * Fill only small enclosed holes inside a binary mask.
     */
    private fun fillSmallHoles(binaryMask: Mat, maxHoleArea: Int): Mat {
        val mask8u = Mat()
        val inv = Mat()
        val flood = Mat()
        val floodMask = Mat.zeros(binaryMask.rows() + 2, binaryMask.cols() + 2, CvType.CV_8U)
        val holes = Mat()
        val result = Mat()

        try {
            Imgproc.threshold(binaryMask, mask8u, 0.0, 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_not(mask8u, inv)
            inv.copyTo(flood)

            Imgproc.floodFill(flood, floodMask, Point(0.0, 0.0), Scalar(128.0))
            Core.compare(flood, Scalar(255.0), holes, Core.CMP_EQ)

            val labels = Mat()
            val stats = Mat()
            val centroids = Mat()

            try {
                val numLabels = Imgproc.connectedComponentsWithStats(
                    holes,
                    labels,
                    stats,
                    centroids,
                    8,
                    CvType.CV_32S
                )

                val smallHoles = Mat.zeros(binaryMask.size(), CvType.CV_8U)
                for (i in 1 until numLabels) {
                    val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
                    if (area <= maxHoleArea) {
                        val componentMask = Mat()
                        Core.compare(labels, Scalar(i.toDouble()), componentMask, Core.CMP_EQ)
                        Core.bitwise_or(smallHoles, componentMask, smallHoles)
                        componentMask.release()
                    }
                }

                Core.bitwise_or(mask8u, smallHoles, result)
                smallHoles.release()
            } finally {
                labels.release()
                stats.release()
                centroids.release()
            }

            return result
        } finally {
            mask8u.release()
            inv.release()
            flood.release()
            floodMask.release()
            holes.release()
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
     * Percentile inside a binary mask for a single-channel Mat.
     */
    private fun percentileInsideMask(src: Mat, mask: Mat, percentile: Double): Double {
        val values = ArrayList<Float>()

        val rows = src.rows()
        val cols = src.cols()

        for (y in 0 until rows) {
            for (x in 0 until cols) {
                val m = mask.get(y, x)?.get(0)?.toInt() ?: 0
                if (m > 0) {
                    val v = src.get(y, x)?.get(0)?.toFloat() ?: 0f
                    values.add(v)
                }
            }
        }

        if (values.isEmpty()) {
            return this.percentile(src, percentile)
        }

        val data = values.toFloatArray()
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
    }

    /**
     * Clamp a float Mat into [minVal, maxVal].
     */
    private fun clampMat(mat: Mat, minVal: Double, maxVal: Double) {
        Core.max(mat, Scalar.all(minVal), mat)
        Core.min(mat, Scalar.all(maxVal), mat)
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