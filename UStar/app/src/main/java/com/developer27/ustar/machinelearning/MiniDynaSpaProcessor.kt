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

object MiniDynaSpaPreprocessor {

    private const val MODEL_NAME = "MiniDynaSpaPreprocessor.ptl"
    private const val INPUT_SIZE = 224   // model input resolution

    private var module: Module? = null   // cached model instance

    // ImageNet normalization used during preprocessing
    private val normMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val normStd = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** Load model once from assets */
    fun load(context: Context, assetName: String = MODEL_NAME) {
        if (module == null) {
            val path = assetFilePath(context, assetName) // copy if needed
            module = Module.load(path)                   // load model
            Log.i("MiniDynaSpa", "✅ Model loaded successfully from $assetName")
        }
    }

    /** Run inference on one bitmap */
    fun run(context: Context, bitmap: Bitmap): Result? {
        if (module == null) load(context)
        val model = module ?: return null

        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        // Convert bitmap -> tensor with normalization
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            normMean,
            normStd
        )

        // Forward pass (expected tuple of 4 tensors)
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

    /** Convert mask tensor [1,1,H,W] -> grayscale bitmap */
    fun maskTensorToBitmap(maskTensor: Tensor): Bitmap {
        val shape = maskTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()

        val data = maskTensor.dataAsFloatArray
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
                val value = (data[index].coerceIn(0f, 1f) * 255f).toInt()
                bitmap.setPixel(x, y, Color.rgb(value, value, value))
                index++
            }
        }
        return bitmap
    }

    /** Convert importance tensor [1,1,H,W] -> grayscale bitmap */
    fun importanceMapToBitmap(importanceTensor: Tensor): Bitmap {
        val shape = importanceTensor.shape()
        val h = shape[2].toInt()
        val w = shape[3].toInt()

        val data = importanceTensor.dataAsFloatArray
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)

        var index = 0
        for (y in 0 until h) {
            for (x in 0 until w) {
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

    /** Copy model file from assets -> internal storage */
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