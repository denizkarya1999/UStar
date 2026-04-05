package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

object MiniDynaSpaPreprocessor {

    private const val MODEL_NAME = "MiniDynaSpaPreprocessor.ptl"
    private const val INPUT_SIZE = 224   // model input resolution

    private var module: Module? = null   // cached model instance

    // ImageNet normalization (must match training)
    private val normMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val normStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun loadModel(context: Context): Module? {
        if (module != null) return module   // reuse if already loaded
        module = LiteModuleLoader.load(assetFilePath(context, MODEL_NAME))
        return module
    }

    fun run(context: Context, bitmap: Bitmap): Result? {
        val model = loadModel(context) ?: return null   // ensure model loaded

        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Convert bitmap -> tensor with normalization
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            normMean,
            normStd
        )

        // Forward pass (returns tuple of 4 tensors)
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

    fun maskTensorToBitmap(maskTensor: Tensor): Bitmap {
        val shape = maskTensor.shape()   // expected: [1, 1, H, W]
        val h = shape[2].toInt()
        val w = shape[3].toInt()

        val data = maskTensor.dataAsFloatArray
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                // convert [0,1] float -> grayscale pixel
                val value = (data[index].coerceIn(0f, 1f) * 255f).toInt()
                bitmap.setPixel(x, y, Color.rgb(value, value, value))
                index++
            }
        }
        return bitmap
    }

    fun importanceMapToBitmap(importanceTensor: Tensor): Bitmap {
        val shape = importanceTensor.shape()   // expected: [1, 1, H, W]
        val h = shape[2].toInt()
        val w = shape[3].toInt()

        val data = importanceTensor.dataAsFloatArray
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                // convert importance map to grayscale visualization
                val value = (data[index].coerceIn(0f, 1f) * 255f).toInt()
                bitmap.setPixel(x, y, Color.rgb(value, value, value))
                index++
            }
        }
        return bitmap
    }

    data class Result(
        val features: Tensor,
        val maskedFeatures: Tensor,
        val mask: Tensor,
        val importanceMap: Tensor
    )

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        // reuse extracted model if already exists
        if (file.exists() && file.length() > 0) return file.absolutePath

        // copy model from assets -> internal storage
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
                output.flush()
            }
        }
        return file.absolutePath
    }
}