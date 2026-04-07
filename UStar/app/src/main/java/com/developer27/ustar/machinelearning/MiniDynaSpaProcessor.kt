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
import kotlin.math.exp

object MiniDynaSpaPreprocessor {

    // TorchScript model that returns (logits, feature_map)
    private const val MODEL_NAME = "UStar_DynaSpa_ResNet50_UOID_Tag_Detection_Mobile.pt"
    private const val INPUT_SIZE = 224

    // Hard-coded class labels
    private val CLASS_LABELS = arrayOf(
        "No UOID tag",
        "UOID tag present"
    )

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

        // Resize input for model
        val processed = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Convert bitmap to tensor
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            processed,
            normMean,
            normStd
        )

        // Model output: (logits, feature_map)
        val outputTuple = model.forward(IValue.from(inputTensor)).toTuple()

        val logitsTensor = outputTuple[0].toTensor()
        val featureMapTensor = outputTuple[1].toTensor()

        val logits = logitsTensor.dataAsFloatArray
        if (logits.isEmpty()) return null

        // Make prediction from logits
        val probabilities = softmax(logits)
        val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[predictedClass]

        // Resolve readable label from hard-coded labels
        val predictedLabel = getLabelForClass(predictedClass)

        return Result(
            logits = logitsTensor,
            featureMap = featureMapTensor,
            predictedClass = predictedClass,
            predictedLabel = predictedLabel,
            confidence = confidence,
            probabilities = probabilities
        )
    }

    // Return label from class id
    fun getLabelForClass(classId: Int): String {
        return if (classId in CLASS_LABELS.indices) {
            CLASS_LABELS[classId]
        } else {
            "Unknown"
        }
    }

    fun featureMapToHeatmapBitmap(
        featureMapTensor: Tensor,
        targetWidth: Int? = null,
        targetHeight: Int? = null
    ): Bitmap {
        val shape = featureMapTensor.shape()

        // Expected shape: [1, C, H, W]
        if (shape.size != 4) {
            throw IllegalArgumentException(
                "Expected feature map shape [1, C, H, W], got ${shape.contentToString()}"
            )
        }

        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        val data = featureMapTensor.dataAsFloatArray

        val spatialMap = FloatArray(height * width)

        // Average channels into one 2D map
        for (c in 0 until channels) {
            val channelOffset = c * height * width
            for (i in 0 until height * width) {
                spatialMap[i] += data[channelOffset + i]
            }
        }

        for (i in spatialMap.indices) {
            spatialMap[i] /= channels.toFloat()
        }

        // Normalize to [0, 1]
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        for (v in spatialMap) {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        var index = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = ((spatialMap[index] - minVal) / range).coerceIn(0f, 1f)
                bitmap.setPixel(x, y, heatColor(value))
                index++
            }
        }

        // Resize back if needed
        return if (targetWidth != null && targetHeight != null) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size)
        var sum = 0f

        for (i in logits.indices) {
            exps[i] = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += exps[i]
        }

        if (sum <= 0f) return FloatArray(logits.size) { 0f }

        for (i in exps.indices) {
            exps[i] /= sum
        }

        return exps
    }

    private fun heatColor(value: Float): Int {
        val v = value.coerceIn(0f, 1f)

        return when {
            v < 0.25f -> {
                val t = v / 0.25f
                Color.rgb(0, (255 * t).toInt(), 255)
            }
            v < 0.5f -> {
                val t = (v - 0.25f) / 0.25f
                Color.rgb(0, 255, (255 * (1f - t)).toInt())
            }
            v < 0.75f -> {
                val t = (v - 0.5f) / 0.25f
                Color.rgb((255 * t).toInt(), 255, 0)
            }
            else -> {
                val t = (v - 0.75f) / 0.25f
                Color.rgb(255, (255 * (1f - t)).toInt(), 0)
            }
        }
    }

    data class Result(
        val logits: Tensor,
        val featureMap: Tensor,
        val predictedClass: Int,
        val predictedLabel: String,
        val confidence: Float,
        val probabilities: FloatArray
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