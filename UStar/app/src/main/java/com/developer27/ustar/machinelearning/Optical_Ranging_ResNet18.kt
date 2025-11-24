package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.roundToInt

class Optical_Ranging_ResNet18 private constructor(private val module: Module) {

    companion object {
        private const val INPUT_SIZE = 224              // CenterCrop(224)
        private const val RESIZE_SHORTER_SIDE = 256     // Resize(256)

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** Load the TorchScript model (.pt) from assets */
        fun loadModel(
            context: Context,
            assetName: String = "UStar_Optical_Ranging_ResNet_18_Simulated.pt"
        ): Optical_Ranging_ResNet18? {
            return try {
                val filePath = assetFilePath(context, assetName)
                val module = Module.load(filePath)
                Log.i("Optical_Ranging_ResNet18", "✅ Model loaded successfully from $assetName")
                Optical_Ranging_ResNet18(module)
            } catch (e: Exception) {
                Log.e("Optical_Ranging_ResNet18", "❌ Failed to load model: ${e.message}", e)
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

    // Labels
    private val classes = arrayOf(
        "1M", "2M", "3M", "4M"
    )

    data class Result(
        val topIndex: Int,
        val topClass: String,
        val probabilities: FloatArray
    )

    /**
     * Preprocess Bitmap to match PyTorch pipeline:
     *   Resize(256) on shorter side + CenterCrop(224)
     */
    private fun preprocessBitmap(src: Bitmap): Bitmap {
        val origW = src.width
        val origH = src.height

        if (origW <= 0 || origH <= 0) {
            return Bitmap.createScaledBitmap(src, INPUT_SIZE, INPUT_SIZE, true)
        }

        // 1) Resize: shorter side -> 256, keep aspect ratio
        val newW: Int
        val newH: Int
        if (origW < origH) {
            // width is shorter → width = 256
            newW = RESIZE_SHORTER_SIDE
            newH = (origH * (RESIZE_SHORTER_SIDE.toFloat() / origW.toFloat())).roundToInt()
        } else {
            // height is shorter → height = 256
            newH = RESIZE_SHORTER_SIDE
            newW = (origW * (RESIZE_SHORTER_SIDE.toFloat() / origH.toFloat())).roundToInt()
        }

        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)

        // 2) Center crop: 224 x 224 from the middle
        val x = ((resized.width - INPUT_SIZE) / 2f).roundToInt().coerceAtLeast(0)
        val y = ((resized.height - INPUT_SIZE) / 2f).roundToInt().coerceAtLeast(0)

        return Bitmap.createBitmap(resized, x, y, INPUT_SIZE, INPUT_SIZE)
    }

    /** Run inference on Bitmap */
    fun run(bitmap: Bitmap): Result {
        // Apply same Resize(256)+CenterCrop(224) as in Colab
        val processed = preprocessBitmap(bitmap)

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(processed, MEAN, STD)
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