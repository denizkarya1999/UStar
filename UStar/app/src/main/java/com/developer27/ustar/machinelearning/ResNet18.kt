package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp

class ResNet18 private constructor(private val module: Module) {

    companion object {
        private const val INPUT_SIZE = 224
        private val MEAN = floatArrayOf(0.4914f, 0.4822f, 0.4465f)
        private val STD = floatArrayOf(0.2023f, 0.1994f, 0.2010f)

        /** Load the TorchScript model (.pt) from assets */
        fun loadModel(
            context: Context,
            assetName: String = "CIFAR_10_ResNet_18.pt"
        ): ResNet18? {
            return try {
                val filePath = assetFilePath(context, assetName)
                val module = Module.load(filePath)
                Log.i("ResNet18", "✅ Model loaded successfully from $assetName")
                ResNet18(module)
            } catch (e: Exception) {
                Log.e("ResNet18", "❌ Failed to load model: ${e.message}", e)
                null
            }
        }

        /** Copy model file from assets to internal storage */
        private fun assetFilePath(context: Context, assetName: String): String {
            val outFile = File(context.filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4096)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                    }
                    output.flush()
                }
            }
            return outFile.absolutePath
        }
    }

    // CIFAR-10 labels
    private val classes = arrayOf(
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    )

    data class Result(
        val topIndex: Int,
        val topClass: String,
        val probabilities: FloatArray
    )

    /** Run inference on Bitmap */
    fun run(bitmap: Bitmap): Result {
        val resized = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE)
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        else bitmap

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resized, MEAN, STD)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

        val logits = outputTensor.dataAsFloatArray
        val probs = softmax(logits)
        val topIdx = argmax(probs)
        val topClass = classes.getOrElse(topIdx) { "class_$topIdx" }

        return Result(topIdx, topClass, probs)
    }

    // Index of max probability
    private fun argmax(arr: FloatArray): Int {
        var best = 0
        for (i in 1 until arr.size) if (arr[i] > arr[best]) best = i
        return best
    }

    // Compute softmax
    private fun softmax(logits: FloatArray): FloatArray {
        var max = logits[0]
        for (i in 1 until logits.size) if (logits[i] > max) max = logits[i]
        var sum = 0.0
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            val v = exp((logits[i] - max).toDouble()).toFloat()
            exps[i] = v
            sum += v
        }
        for (i in exps.indices) exps[i] = (exps[i] / sum).toFloat()
        return exps
    }
}