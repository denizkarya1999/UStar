package com.developer27.ustar.machinelearning.DynaSpa

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object DynaSpaMaskProcessor {

    private const val DYNASPA_THRESHOLD = 0.70
    private const val USE_SOFT_DYNASPA = true

    /**
     * Uses the already-created bounding box:
     * crop -> DynaSpa mask -> reconstruct on black background
     */
    fun processBoundingBox(
        source: Bitmap,
        boundingBox: Rect
    ): Bitmap {
        val config = source.config ?: Bitmap.Config.ARGB_8888

        // Black output canvas; only masked bbox content will be restored
        val resultBitmap = Bitmap.createBitmap(source.width, source.height, config)
        resultBitmap.eraseColor(Color.BLACK)

        // Clamp bbox to valid image bounds
        val safeRect = Rect(
            boundingBox.left.coerceIn(0, source.width - 1),
            boundingBox.top.coerceIn(0, source.height - 1),
            boundingBox.right.coerceIn(1, source.width),
            boundingBox.bottom.coerceIn(1, source.height)
        )

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

        // Apply DynaSpa only inside the crop
        val processedCrop = applyImprovedDynaSpaMaskToCrop(
            input = croppedBitmap,
            threshold = DYNASPA_THRESHOLD,
            softMask = USE_SOFT_DYNASPA
        )

        // Put the processed crop back in its original location
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(processedCrop, safeRect.left.toFloat(), safeRect.top.toFloat(), null)

        return resultBitmap
    }

    /**
     * Builds the DynaSpa mask for the cropped bbox region
     * and applies it to the crop.
     */
    private fun applyImprovedDynaSpaMaskToCrop(
        input: Bitmap,
        threshold: Double,
        softMask: Boolean
    ): Bitmap {
        val rgba = Mat()
        val rgb = Mat()

        try {
            Utils.bitmapToMat(input, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

            // Create soft or hard structural mask for the crop
            val dynaspaMask = createImprovedDynaSpaMaskFromRegion(
                regionRgb = rgb,
                threshold = threshold,
                softMask = softMask
            )

            val result = Mat.zeros(rgb.size(), CvType.CV_8UC3)

            try {
                if (softMask) {
                    // Soft mask: preserve weighted structure instead of binary keep/remove
                    val mask3 = Mat()
                    val rgbFloat = Mat()
                    val maskFloat = Mat()
                    val maskedFloat = Mat()

                    try {
                        Core.merge(listOf(dynaspaMask, dynaspaMask, dynaspaMask), mask3)

                        rgb.convertTo(rgbFloat, CvType.CV_32FC3)
                        mask3.convertTo(maskFloat, CvType.CV_32FC3)

                        Core.multiply(rgbFloat, maskFloat, maskedFloat)

                        // Slight brightness boost like the Python version
                        Core.multiply(maskedFloat, Scalar(1.35, 1.35, 1.35), maskedFloat)
                        Core.min(maskedFloat, Scalar(255.0, 255.0, 255.0), maskedFloat)

                        maskedFloat.convertTo(result, CvType.CV_8UC3)
                    } finally {
                        mask3.release()
                        rgbFloat.release()
                        maskFloat.release()
                        maskedFloat.release()
                    }
                } else {
                    // Hard mask: keep only binary-selected structure
                    val binaryMask = Mat()
                    try {
                        dynaspaMask.convertTo(binaryMask, CvType.CV_8UC1, 255.0)
                        rgb.copyTo(result, binaryMask)
                    } finally {
                        binaryMask.release()
                    }
                }

                val resultRgba = Mat()
                try {
                    Imgproc.cvtColor(result, resultRgba, Imgproc.COLOR_RGB2RGBA)
                    val outputBitmap = Bitmap.createBitmap(
                        result.cols(),
                        result.rows(),
                        Bitmap.Config.ARGB_8888
                    )
                    Utils.matToBitmap(resultRgba, outputBitmap)
                    return outputBitmap
                } finally {
                    resultRgba.release()
                }
            } finally {
                dynaspaMask.release()
                result.release()
            }
        } finally {
            rgba.release()
            rgb.release()
        }
    }

    /**
     * OpenCV version of the Python DynaSpa crop mask:
     * enhancement -> cue fusion -> cleanup -> soft/hard mask
     */
    private fun createImprovedDynaSpaMaskFromRegion(
        regionRgb: Mat,
        threshold: Double,
        softMask: Boolean
    ): Mat {
        val enhanced = enhanceRegionForTag(regionRgb)

        val regionBgr = Mat()
        val lab = Mat()
        val hsv = Mat()

        try {
            Imgproc.cvtColor(enhanced, regionBgr, Imgproc.COLOR_RGB2BGR)
            Imgproc.cvtColor(regionBgr, lab, Imgproc.COLOR_BGR2Lab)
            Imgproc.cvtColor(regionBgr, hsv, Imgproc.COLOR_BGR2HSV)

            val labChannels = ArrayList<Mat>(3)
            val hsvChannels = ArrayList<Mat>(3)

            Core.split(lab, labChannels)
            Core.split(hsv, hsvChannels)

            try {
                val lChan = labChannels[0]
                val sChan = hsvChannels[1]
                val vChan = hsvChannels[2]

                // Improve luminance contrast
                val clahe = Imgproc.createCLAHE(2.5, Size(8.0, 8.0))
                val lEq = Mat()
                clahe.apply(lChan, lEq)

                // Smooth while keeping useful structure
                val lSmooth = Mat()
                val sSmooth = Mat()
                Imgproc.bilateralFilter(lEq, lSmooth, 7, 60.0, 60.0)
                Imgproc.bilateralFilter(sChan, sSmooth, 7, 60.0, 60.0)

                // Structural detail cues
                val lap = Mat()
                Imgproc.Laplacian(lSmooth, lap, CvType.CV_32F, 3)
                val lapAbs = Mat()
                Core.absdiff(lap, Scalar.all(0.0), lapAbs)

                val canny = Mat()
                Imgproc.Canny(lSmooth, canny, 40.0, 120.0)

                val lapNorm = normalizeMask(lapAbs)
                val lNorm = normalizeMask(lSmooth)
                val sNorm = normalizeMask(sSmooth)
                val vNorm = normalizeMask(vChan)

                val cannyFloat = Mat()
                canny.convertTo(cannyFloat, CvType.CV_32F, 1.0 / 255.0)

                try {
                    // Weighted fusion of luminance, saturation, Laplacian, and Canny
                    val combined = Mat.zeros(regionRgb.size(), CvType.CV_32F)
                    val temp = Mat()

                    try {
                        Core.addWeighted(lNorm, 0.35, vNorm, 0.20, 0.0, combined)
                        Core.addWeighted(sNorm, 0.20, lapNorm, 0.15, 0.0, temp)
                        Core.add(combined, temp, combined)

                        Core.multiply(cannyFloat, Scalar(0.10), temp)
                        Core.add(combined, temp, combined)

                        val combinedNorm = normalizeMask(combined)

                        // Strengthen structural regions a bit more
                        val maxStruct = Mat()
                        val structureBoost = Mat()

                        try {
                            Core.max(lapNorm, cannyFloat, maxStruct)
                            Core.addWeighted(combinedNorm, 0.7, maxStruct, 0.3, 0.0, structureBoost)
                            Imgproc.GaussianBlur(structureBoost, structureBoost, Size(0.0, 0.0), 1.2, 1.2)

                            val structureBoostNorm = normalizeMask(structureBoost)

                            // Convert importance map into binary region
                            val binaryFloat = Mat()
                            Imgproc.threshold(
                                structureBoostNorm,
                                binaryFloat,
                                threshold,
                                1.0,
                                Imgproc.THRESH_BINARY
                            )

                            val binary = Mat()
                            binaryFloat.convertTo(binary, CvType.CV_8U, 255.0)

                            try {
                                // Morphological cleanup
                                val kernelOpen = Imgproc.getStructuringElement(
                                    Imgproc.MORPH_ELLIPSE,
                                    Size(3.0, 3.0)
                                )
                                val kernelClose = Imgproc.getStructuringElement(
                                    Imgproc.MORPH_ELLIPSE,
                                    Size(5.0, 5.0)
                                )
                                val kernelDilate = Imgproc.getStructuringElement(
                                    Imgproc.MORPH_ELLIPSE,
                                    Size(3.0, 3.0)
                                )

                                try {
                                    Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernelOpen)
                                    Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernelClose)

                                    val cleaned = keepTopConnectedComponents(
                                        binaryMask = binary,
                                        maxComponents = 10,
                                        minArea = 12
                                    )

                                    try {
                                        Imgproc.dilate(cleaned, cleaned, kernelDilate)

                                        return if (softMask) {
                                            // Soft mask keeps weighted confidence inside cleaned region
                                            val cleanedFloat = Mat()
                                            val soft = Mat()

                                            try {
                                                cleaned.convertTo(cleanedFloat, CvType.CV_32F, 1.0 / 255.0)
                                                Core.multiply(structureBoostNorm, cleanedFloat, soft)
                                                Imgproc.GaussianBlur(soft, soft, Size(0.0, 0.0), 1.0, 1.0)
                                                normalizeMask(soft)
                                            } finally {
                                                cleanedFloat.release()
                                                soft.release()
                                            }
                                        } else {
                                            // Hard mask returns 0/1 region
                                            val hard = Mat()
                                            try {
                                                cleaned.convertTo(hard, CvType.CV_32F, 1.0 / 255.0)
                                                hard.clone()
                                            } finally {
                                                hard.release()
                                            }
                                        }
                                    } finally {
                                        cleaned.release()
                                    }
                                } finally {
                                    kernelOpen.release()
                                    kernelClose.release()
                                    kernelDilate.release()
                                }
                            } finally {
                                binaryFloat.release()
                                binary.release()
                            }
                        } finally {
                            maxStruct.release()
                            structureBoost.release()
                        }
                    } finally {
                        combined.release()
                        temp.release()
                    }
                } finally {
                    cannyFloat.release()
                    lapNorm.release()
                    lNorm.release()
                    sNorm.release()
                    vNorm.release()
                    lEq.release()
                    lSmooth.release()
                    sSmooth.release()
                    lap.release()
                    lapAbs.release()
                    canny.release()
                }
            } finally {
                labChannels.forEach { it.release() }
                hsvChannels.forEach { it.release() }
            }
        } finally {
            enhanced.release()
            regionBgr.release()
            lab.release()
            hsv.release()
        }
    }

    /**
     * Light pre-enhancement for the crop.
     */
    private fun enhanceRegionForTag(regionRgb: Mat): Mat {
        val regionBgr = Mat()
        val den = Mat()
        val lab = Mat()
        val enhancedBgr = Mat()
        val enhancedRgb = Mat()

        try {
            Imgproc.cvtColor(regionRgb, regionBgr, Imgproc.COLOR_RGB2BGR)

            // Bilateral filter reduces noise while preserving edges
            Imgproc.bilateralFilter(regionBgr, den, 7, 50.0, 50.0)

            // CLAHE on luminance channel
            Imgproc.cvtColor(den, lab, Imgproc.COLOR_BGR2Lab)
            val labChannels = ArrayList<Mat>(3)
            Core.split(lab, labChannels)

            try {
                val clahe = Imgproc.createCLAHE(2.5, Size(8.0, 8.0))
                clahe.apply(labChannels[0], labChannels[0])

                Core.merge(labChannels, lab)
                Imgproc.cvtColor(lab, enhancedBgr, Imgproc.COLOR_Lab2BGR)
                Imgproc.cvtColor(enhancedBgr, enhancedRgb, Imgproc.COLOR_BGR2RGB)

                return enhancedRgb.clone()
            } finally {
                labChannels.forEach { it.release() }
            }
        } finally {
            regionBgr.release()
            den.release()
            lab.release()
            enhancedBgr.release()
            enhancedRgb.release()
        }
    }

    /**
     * Normalize single-channel Mat into [0,1].
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
     * Remove tiny disconnected regions and keep only the largest ones.
     */
    private fun keepTopConnectedComponents(
        binaryMask: Mat,
        maxComponents: Int = 10,
        minArea: Int = 12
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

            val components = mutableListOf<Pair<Int, Int>>()

            for (i in 1 until numLabels) {
                val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
                if (area >= minArea) {
                    components.add(i to area)
                }
            }

            val keepIds = components
                .sortedByDescending { it.second }
                .take(maxComponents)
                .map { it.first }
                .toSet()

            val cleaned = Mat.zeros(binaryMask.size(), CvType.CV_8U)

            for (y in 0 until labels.rows()) {
                for (x in 0 until labels.cols()) {
                    val label = labels.get(y, x)[0].toInt()
                    if (label in keepIds) {
                        cleaned.put(y, x, 255.0)
                    }
                }
            }

            return cleaned
        } finally {
            labels.release()
            stats.release()
            centroids.release()
        }
    }
}