package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

object MiniDynaSpaPreprocessor {

    private const val MODEL_NAME = "MiniDynaSpaPreprocessor.ptl"
    private const val INPUT_SIZE = 224

    private var module: Module? = null

    // ImageNet normalization
    private val normMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val normStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun load(context: Context, assetName: String = MODEL_NAME) {
        if (module == null) {
            val path = assetFilePath(context, assetName)
            module = Module.load(path)
            Log.i("MiniDynaSpa", "Model loaded successfully from $assetName")
        }
    }

    fun run(context: Context, bitmap: Bitmap): Result? {
        if (module == null) load(context)
        val model = module ?: return null

        // Resize like the notebook input pipeline
        val processed = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            processed,
            normMean,
            normStd
        )

        val outputTuple = model.forward(IValue.from(inputTensor)).toTuple()

        val featuresTensor = outputTuple[0].toTensor()
        val maskedFeaturesTensor = outputTuple[1].toTensor()
        val maskTensor = outputTuple[2].toTensor()
        val importanceTensor = outputTuple[3].toTensor()

        return Result(
            features = featuresTensor,
            maskedFeatures = maskedFeaturesTensor,
            mask = maskTensor,
            importanceMap = importanceTensor
        )
    }

    /**
     * Better masked-image visualization for RGB images.
     *
     * Why this works better:
     * - keeps background slightly visible
     * - preserves original colors
     * - makes important regions brighter/clearer
     */
    fun applyMaskToOriginal(original: Bitmap, maskTensor: Tensor): Bitmap {
        val maskBitmap = softMaskTensorToBitmap(
            maskTensor = maskTensor,
            targetWidth = original.width,
            targetHeight = original.height
        )

        val output = Bitmap.createBitmap(
            original.width,
            original.height,
            Bitmap.Config.ARGB_8888
        )

        val originalPixels = IntArray(original.width * original.height)
        val maskPixels = IntArray(original.width * original.height)
        val outPixels = IntArray(original.width * original.height)

        original.getPixels(
            originalPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        maskBitmap.getPixels(
            maskPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        for (i in originalPixels.indices) {
            val pixel = originalPixels[i]
            val maskGray = maskPixels[i] and 0xFF
            val norm = (maskGray / 255f).coerceIn(0f, 1f)

            // Stronger focus on important regions
            val boosted = smoothPower(norm, 1.35f)

            // Keep some visibility in the background for RGB images
            val visibility = 0.38f + 0.62f * boosted

            // Slight highlight for important pixels
            val highlight = 1.0f + 0.20f * boosted

            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            val outR = (r * visibility * highlight).toInt().coerceIn(0, 255)
            val outG = (g * visibility * highlight).toInt().coerceIn(0, 255)
            val outB = (b * visibility * highlight).toInt().coerceIn(0, 255)

            outPixels[i] = (0xFF shl 24) or
                    (outR shl 16) or
                    (outG shl 8) or
                    outB
        }

        output.setPixels(
            outPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        return output
    }

    /**
     * Optional overlay version.
     * This is often even better for RGB because it keeps the full image visible.
     */
    fun applyHeatmapOverlayToOriginal(original: Bitmap, maskTensor: Tensor): Bitmap {
        val overlay = attentionMaskTensorToBitmap(
            maskTensor = maskTensor,
            targetWidth = original.width,
            targetHeight = original.height
        )

        val output = Bitmap.createBitmap(
            original.width,
            original.height,
            Bitmap.Config.ARGB_8888
        )

        val originalPixels = IntArray(original.width * original.height)
        val overlayPixels = IntArray(original.width * original.height)
        val outPixels = IntArray(original.width * original.height)

        original.getPixels(
            originalPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        overlay.getPixels(
            overlayPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        for (i in originalPixels.indices) {
            val base = originalPixels[i]
            val over = overlayPixels[i]

            val br = (base shr 16) and 0xFF
            val bg = (base shr 8) and 0xFF
            val bb = base and 0xFF

            val orr = (over shr 16) and 0xFF
            val og = (over shr 8) and 0xFF
            val ob = over and 0xFF

            // Use mask intensity to control overlay strength
            val avgOverlay = ((orr + og + ob) / 3f) / 255f
            val alpha = 0.12f + 0.33f * avgOverlay

            val r = (br * (1f - alpha) + orr * alpha).toInt().coerceIn(0, 255)
            val g = (bg * (1f - alpha) + og * alpha).toInt().coerceIn(0, 255)
            val b = (bb * (1f - alpha) + ob * alpha).toInt().coerceIn(0, 255)

            outPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        output.setPixels(
            outPixels,
            0,
            original.width,
            0,
            0,
            original.width,
            original.height
        )

        return output
    }

    fun softMaskTensorToBitmap(
        maskTensor: Tensor,
        targetWidth: Int? = null,
        targetHeight: Int? = null
    ): Bitmap {
        val shape = maskTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val data = maskTensor.dataAsFloatArray

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val minVal = data.minOrNull() ?: 0f
        val maxVal = data.maxOrNull() ?: 1f
        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val norm = ((data[index] - minVal) / range).coerceIn(0f, 1f)

                // Smooth mask for better resized output
                val boosted = smoothPower(norm, 1.2f)
                val value = (boosted * 255f).toInt().coerceIn(0, 255)

                bitmap.setPixel(x, y, Color.rgb(value, value, value))
                index++
            }
        }

        return if (targetWidth != null && targetHeight != null) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
    }

    fun attentionMaskTensorToBitmap(
        maskTensor: Tensor,
        targetWidth: Int? = null,
        targetHeight: Int? = null
    ): Bitmap {
        val shape = maskTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val data = maskTensor.dataAsFloatArray

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val minVal = data.minOrNull() ?: 0f
        val maxVal = data.maxOrNull() ?: 1f
        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val norm = ((data[index] - minVal) / range).coerceIn(0f, 1f)
                val boosted = smoothPower(norm, 1.9f)
                bitmap.setPixel(x, y, warmGlowColor(boosted))
                index++
            }
        }

        return if (targetWidth != null && targetHeight != null) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
    }

    fun importanceMapToBitmap(
        importanceTensor: Tensor,
        targetWidth: Int? = null,
        targetHeight: Int? = null
    ): Bitmap {
        val shape = importanceTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val data = importanceTensor.dataAsFloatArray

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        val minVal = data.minOrNull() ?: 0f
        val maxVal = data.maxOrNull() ?: 1f
        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val norm = ((data[index] - minVal) / range).coerceIn(0f, 1f)
                val boosted = smoothPower(norm, 1.8f)
                bitmap.setPixel(x, y, warmGlowColor(boosted))
                index++
            }
        }

        return if (targetWidth != null && targetHeight != null) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
    }

    data class Result(
        val features: Tensor,
        val maskedFeatures: Tensor,
        val mask: Tensor,
        val importanceMap: Tensor
    )

    // Warm colors for the attention map
    private fun warmGlowColor(v: Float): Int {
        val x = v.coerceIn(0f, 1f)

        return when {
            x < 0.15f -> {
                val t = x / 0.15f
                val c = (t * 40f).toInt().coerceIn(0, 40)
                Color.rgb(c / 3, c / 3, c / 4)
            }
            x < 0.40f -> {
                val t = (x - 0.15f) / 0.25f
                val r = lerp(25, 120, t)
                val g = lerp(22, 95, t)
                val b = lerp(18, 55, t)
                Color.rgb(r, g, b)
            }
            x < 0.70f -> {
                val t = (x - 0.40f) / 0.30f
                val r = lerp(120, 255, t)
                val g = lerp(95, 210, t)
                val b = lerp(55, 150, t)
                Color.rgb(r, g, b)
            }
            else -> {
                val t = (x - 0.70f) / 0.30f
                val r = lerp(255, 255, t)
                val g = lerp(210, 245, t)
                val b = lerp(150, 210, t)
                Color.rgb(r, g, b)
            }
        }
    }

    private fun lerp(a: Int, b: Int, t: Float): Int {
        val tt = t.coerceIn(0f, 1f)
        return (a + (b - a) * tt).toInt().coerceIn(0, 255)
    }

    private fun smoothPower(x: Float, gamma: Float): Float {
        return x.coerceIn(0f, 1f).let { v ->
            max(0f, min(1f, Math.pow(v.toDouble(), gamma.toDouble()).toFloat()))
        }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4096)
                while (true) {
                    val read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        return file.absolutePath
    }
}