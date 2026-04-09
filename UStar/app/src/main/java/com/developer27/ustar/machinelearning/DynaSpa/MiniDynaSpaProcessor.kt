package com.developer27.ustar.machinelearning.DynaSpa

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
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

    // Sensitivity to "heat" for bounding box extraction (0.0 to 1.0)
    // Lower = larger box, higher = tighter box
    var globalThresholdRatio = 0.70f

    // Hard-coded class labels
    const val CLASS_NO_UOID = 0
    const val CLASS_UOID_PRESENT = 1

    private val CLASS_LABELS = arrayOf(
        "No UOID Tag",
        "UOID Tag Present"
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

        // Resize for model input
        val processed = resize(bitmap)

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

        // Convert logits to probabilities
        val probabilities = softmax(logits)
        val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[predictedClass]
        val predictedLabel = getLabelForClass(predictedClass)

        return Result(
            processedBitmap = processed,   // keep processed image for display/masking
            logits = logitsTensor,
            featureMap = featureMapTensor,
            predictedClass = predictedClass,
            predictedLabel = predictedLabel,
            confidence = confidence,
            probabilities = probabilities
        )
    }

    // Resize image to model input size (224x224)
    // If input is larger than 256x256, crop 1024x1024 (or max available square) from middle first, then resize to 224x224
    private fun resize(src: Bitmap): Bitmap {
        val targetInputSize = 224
        
        return if (src.width > 256 || src.height > 256) {
            val intermediateCropSize = 1024
            val cropSize = minOf(minOf(src.width, src.height), intermediateCropSize)
            
            val left = (src.width - cropSize) / 2
            val top = (src.height - cropSize) / 2
            
            val cropped = Bitmap.createBitmap(src, left, top, cropSize, cropSize)
            Bitmap.createScaledBitmap(cropped, targetInputSize, targetInputSize, true)
        } else {
            // If 256x256 or smaller, just resize
            Bitmap.createScaledBitmap(src, targetInputSize, targetInputSize, true)
        }
    }

    // Return label from class id
    fun getLabelForClass(classId: Int): String {
        return if (classId in CLASS_LABELS.indices) {
            CLASS_LABELS[classId]
        } else {
            "Unknown"
        }
    }

    fun featureMapToHeatmapBitmap(featureMapTensor: Tensor): Bitmap {
        val spatialMap = featureMapToSpatialMap(featureMapTensor)
        val height = spatialMap.size
        val width = spatialMap[0].size

        // Find min/max for normalization
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = spatialMap[y][x]
                if (v < minVal) minVal = v
                if (v > maxVal) maxVal = v
            }
        }

        val range = if (maxVal - minVal < 1e-8f) 1f else (maxVal - minVal)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Convert normalized values to heat colors
        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = ((spatialMap[y][x] - minVal) / range).coerceIn(0f, 1f)
                bitmap.setPixel(x, y, heatColor(v))
            }
        }

        return bitmap
    }

    fun shouldShowBoundingBox(predictedClass: Int): Boolean {
        return predictedClass == CLASS_UOID_PRESENT
    }

    // Create bounding-box masked bitmap from feature-map intensity
    fun featureMapToBoundingBoxMaskedBitmap(
        processedBitmap: Bitmap,
        featureMapTensor: Tensor,
        predictedClass: Int,
        thresholdRatio: Float = globalThresholdRatio
    ): Bitmap {
        // Return black image if UOID tag is not predicted
        if (!shouldShowBoundingBox(predictedClass)) {
            return Bitmap.createBitmap(
                processedBitmap.width,
                processedBitmap.height,
                Bitmap.Config.ARGB_8888
            ).apply { eraseColor(Color.BLACK) }
        }

        val box = extractBoundingBoxFromFeatureMap(
            featureMapTensor = featureMapTensor,
            outputWidth = processedBitmap.width,
            outputHeight = processedBitmap.height,
            thresholdRatio = thresholdRatio
        )

        val output = Bitmap.createBitmap(
            processedBitmap.width,
            processedBitmap.height,
            Bitmap.Config.ARGB_8888
        )

        val inputPixels = IntArray(processedBitmap.width * processedBitmap.height)
        val outputPixels = IntArray(processedBitmap.width * processedBitmap.height)

        processedBitmap.getPixels(
            inputPixels,
            0,
            processedBitmap.width,
            0,
            0,
            processedBitmap.width,
            processedBitmap.height
        )

        // Keep only box region, black out the rest
        for (y in 0 until processedBitmap.height) {
            for (x in 0 until processedBitmap.width) {
                val index = y * processedBitmap.width + x
                outputPixels[index] = if (box.contains(x, y)) {
                    inputPixels[index]
                } else {
                    Color.BLACK
                }
            }
        }

        output.setPixels(
            outputPixels,
            0,
            processedBitmap.width,
            0,
            0,
            processedBitmap.width,
            processedBitmap.height
        )

        return output
    }

    // Extract bounding box from high-intensity feature-map region
    fun extractBoundingBoxFromFeatureMap(
        featureMapTensor: Tensor,
        outputWidth: Int,
        outputHeight: Int,
        thresholdRatio: Float = globalThresholdRatio
    ): Rect {
        val spatialMap = featureMapToSpatialMap(featureMapTensor)
        val mapHeight = spatialMap.size
        val mapWidth = spatialMap[0].size

        // Find max intensity
        var maxVal = -Float.MAX_VALUE
        for (y in 0 until mapHeight) {
            for (x in 0 until mapWidth) {
                if (spatialMap[y][x] > maxVal) maxVal = spatialMap[y][x]
            }
        }

        val threshold = maxVal * thresholdRatio.coerceIn(0f, 1f)

        var minX = mapWidth
        var minY = mapHeight
        var maxX = -1
        var maxY = -1

        // Collect high-activation region
        for (y in 0 until mapHeight) {
            for (x in 0 until mapWidth) {
                if (spatialMap[y][x] >= threshold) {
                    if (x < minX) minX = x
                    if (y < minY) minY = y
                    if (x > maxX) maxX = x
                    if (y > maxY) maxY = y
                }
            }
        }

        // Fallback to full image if nothing passes threshold
        if (maxX < minX || maxY < minY) {
            return Rect(0, 0, outputWidth, outputHeight)
        }

        // Map feature-map coordinates to processed image coordinates
        val left = (minX.toFloat() / mapWidth * outputWidth).toInt().coerceIn(0, outputWidth - 1)
        val top = (minY.toFloat() / mapHeight * outputHeight).toInt().coerceIn(0, outputHeight - 1)
        val right = ((maxX + 1).toFloat() / mapWidth * outputWidth).toInt().coerceIn(left + 1, outputWidth)
        val bottom = ((maxY + 1).toFloat() / mapHeight * outputHeight).toInt().coerceIn(top + 1, outputHeight)

        return Rect(left, top, right, bottom)
    }

    // Convert feature map [1,C,H,W] to averaged 2D spatial map
    private fun featureMapToSpatialMap(featureMapTensor: Tensor): Array<FloatArray> {
        val shape = featureMapTensor.shape()

        // Expected [1, C, H, W]
        if (shape.size != 4) {
            throw IllegalArgumentException(
                "Expected feature map shape [1, C, H, W], got ${shape.contentToString()}"
            )
        }

        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        val data = featureMapTensor.dataAsFloatArray

        val spatialMap = Array(height) { FloatArray(width) }

        // Average channels into one map
        for (c in 0 until channels) {
            val offset = c * height * width
            for (y in 0 until height) {
                for (x in 0 until width) {
                    spatialMap[y][x] += data[offset + y * width + x]
                }
            }
        }

        for (y in 0 until height) {
            for (x in 0 until width) {
                spatialMap[y][x] /= channels.toFloat()
            }
        }

        return spatialMap
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size)
        var sum = 0f

        // Stable softmax
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
        val processedBitmap: Bitmap,
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