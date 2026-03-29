package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.random.Random

class Orientation_Guidance_ResNet18 private constructor(private val module: Module) {

    companion object {
        private const val INPUT_WIDTH = 256   // model input width
        private const val INPUT_HEIGHT = 256  // model input height

        // ImageNet normalization (must match training)
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        /** Load TorchScript model from assets */
        fun loadModel(
            context: Context,
            assetName: String = "UStar_Orientation_Guidance_ResNet_18_Simulated_Images_Ahmad.pt"
        ): Orientation_Guidance_ResNet18? {
            return try {
                val filePath = assetFilePath(context, assetName) // copy if needed
                val module = Module.load(filePath)               // load .pt model
                Log.i("Orientation_Guidance_ResNet18", "✅ Model loaded successfully")
                Orientation_Guidance_ResNet18(module)
            } catch (e: Exception) {
                Log.e("Orientation_Guidance_ResNet18", "❌ Failed to load model: ${e.message}", e)
                null
            }
        }

        /** Copy asset → internal storage (required by PyTorch) */
        private fun assetFilePath(context: Context, assetName: String): String {
            val outFile = File(context.filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4096) // copy buffer
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                    }
                    output.flush() // ensure write completes
                }
            }
            return outFile.absolutePath
        }
    }

    // Class labels (order must match training dataset)
    private val classes = arrayOf(
        "East", "North", "Northeast", "Northwest",
        "South", "Southeast", "Southwest", "West"
    )

    data class Result(
        val topIndex: Int,         // predicted class index
        val topClass: String,      // predicted class name
        val probabilities: FloatArray // softmax probabilities
    )

    /** Resize input to fixed size (256×256) */
    private fun preprocessBitmap(src: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(src, INPUT_WIDTH, INPUT_HEIGHT, true) // resize only
    }

    /** Optional debug augmentation (Rotation + vertical shift) */
    /** Equivalent to PyTorch RandomAffine(degrees=360, translate=(0.0, 0.3), fill=0) */
    private fun applyRandomAffineDebug(src: Bitmap): Bitmap {
        val w = src.width.toFloat()
        val h = src.height.toFloat()

        // random rotation: 0–360°
        val angle = Random.nextFloat() * 360f

        // random vertical shift: ±30% of height
        val maxDy = 0.3f * h
        val dy = Random.nextFloat() * 2f * maxDy - maxDy

        val matrix = Matrix()

        // rotate around center
        matrix.postRotate(angle, w / 2f, h / 2f)

        // translate vertically
        matrix.postTranslate(0f, dy)

        // apply transform (black fill by default)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    /** Run inference */
    fun run(bitmap: Bitmap): Result {
        // step 1: resize
        val processed = preprocessBitmap(bitmap)

        // optional: apply 360° rotation + vertical shift for augmentation (debugging purposes only).
        //val processed_rotated = applyRandomAffineDebug(processed)

        // step 2: convert Bitmap → tensor + normalize
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(processed, MEAN, STD)

        // step 3: forward pass through model
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()

        // step 4: extract raw logits
        val logits = outputTensor.dataAsFloatArray

        // step 5: convert logits → probabilities
        val probs = softmax(logits)

        // step 6: pick highest probability class
        val topIdx = argmax(probs)
        val topClass = classes.getOrElse(topIdx) { "class_$topIdx" }

        return Result(topIdx, topClass, probs)
    }

    /** Find index of maximum value */
    private fun argmax(arr: FloatArray): Int {
        var bestIdx = 0
        var bestVal = arr[0]
        for (i in 1 until arr.size) {
            if (arr[i] > bestVal) {
                bestVal = arr[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    /** Softmax: converts logits → probabilities */
    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = logits[0]

        // find max (for numerical stability)
        for (i in 1 until logits.size) {
            if (logits[i] > maxLogit) maxLogit = logits[i]
        }

        var sum = 0.0
        val exps = FloatArray(logits.size)

        // exponentiate shifted logits
        for (i in logits.indices) {
            val v = exp((logits[i] - maxLogit).toDouble()).toFloat()
            exps[i] = v
            sum += v
        }

        // normalize to probabilities
        for (i in exps.indices) {
            exps[i] = (exps[i] / sum).toFloat()
        }

        return exps
    }
}