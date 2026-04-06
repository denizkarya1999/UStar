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

object MiniDynaSpaPreprocessor {

    // Changed from .ptl to .pt
    private const val MODEL_NAME = "UStar_MiniDynaSpa_Denoising.pt"
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
     * Hard black-background masking like Colab.
     * Keeps only the top maskRate strongest regions.
     *
     * Example:
     * maskRate = 0.06f keeps about top 6% strongest mask pixels.
     */
    fun applyHardMaskToOriginal(
        original: Bitmap,
        maskTensor: Tensor,
        maskRate: Float = 0.06f
    ): Bitmap {
        val binaryMaskBitmap = hardMaskTensorToBitmap(
            maskTensor = maskTensor,
            targetWidth = original.width,
            targetHeight = original.height,
            maskRate = maskRate
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

        binaryMaskBitmap.getPixels(
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
            val keep = if (maskGray > 0) 1f else 0f

            val r = ((pixel shr 16) and 0xFF) * keep
            val g = ((pixel shr 8) and 0xFF) * keep
            val b = (pixel and 0xFF) * keep

            outPixels[i] = (0xFF shl 24) or
                    (r.toInt().coerceIn(0, 255) shl 16) or
                    (g.toInt().coerceIn(0, 255) shl 8) or
                    b.toInt().coerceIn(0, 255)
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
     * Converts mask tensor to a hard binary bitmap using top-k masking.
     * This matches the Colab logic more closely.
     */
    fun hardMaskTensorToBitmap(
        maskTensor: Tensor,
        targetWidth: Int? = null,
        targetHeight: Int? = null,
        maskRate: Float = 0.06f
    ): Bitmap {
        val shape = maskTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()
        val data = maskTensor.dataAsFloatArray.copyOf()

        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        // Normalize to [0, 1]
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        for (v in data) {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }
        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)

        val normalized = FloatArray(data.size)
        for (i in data.indices) {
            normalized[i] = ((data[i] - minVal) / range).coerceIn(0f, 1f)
        }

        // Top-k threshold
        val safeMaskRate = maskRate.coerceIn(0f, 1f)
        val total = normalized.size
        val k = max(1, (total * safeMaskRate).toInt())

        val sorted = normalized.copyOf()
        sorted.sort()
        val threshold = sorted[max(0, total - k)]

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val keep = normalized[index] >= threshold
                val value = if (keep) 255 else 0
                bitmap.setPixel(x, y, Color.rgb(value, value, value))
                index++
            }
        }

        return if (targetWidth != null && targetHeight != null) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, false)
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