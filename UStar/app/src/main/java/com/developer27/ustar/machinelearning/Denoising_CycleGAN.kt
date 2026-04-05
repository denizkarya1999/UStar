package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.roundToInt

object Denoising_CycleGAN {

    private var module: Module? = null
    private const val IMG = 256 // model input size

    /** Load model once from assets */
    fun load(context: Context, assetName: String = "UStar_Denoising_CycleGAN.ptl") {
        if (module == null) {
            val path = assetFilePath(context, assetName) // copy if needed
            module = Module.load(path)                   // load model
            Log.i("Denoising_CycleGAN", "✅ Model loaded successfully from $assetName")
        }
    }

    /** Run inference on input image */
    fun run(input: Bitmap): Bitmap {
        val pre = centerCropResizeToSquare(input, IMG, 1.12f) // preprocess

        // convert Bitmap → tensor + normalize [-1,1]
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            pre,
            floatArrayOf(0.5f, 0.5f, 0.5f), // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)  // std
        )

        // forward pass
        val outTensor = module!!.forward(IValue.from(tensor)).toTensor()
        val outArray = outTensor.dataAsFloatArray // NCHW output

        // convert output tensor → Bitmap
        return floatArrayToBitmap(
            outArray,
            IMG,
            IMG,
            floatArrayOf(0.5f, 0.5f, 0.5f),
            floatArrayOf(0.5f, 0.5f, 0.5f)
        )
    }

    /** Resize (short side) + center crop to square */
    private fun centerCropResizeToSquare(input: Bitmap, size: Int, scaleMult: Float): Bitmap {
        val inW = input.width
        val inH = input.height
        val target = (size * scaleMult).toInt() // scaled short side

        // compute scale factor
        val scale = if (inW < inH) target.toFloat() / inW else target.toFloat() / inH

        val scaledW = (inW * scale).roundToInt()
        val scaledH = (inH * scale).roundToInt()

        val resized = Bitmap.createScaledBitmap(input, scaledW, scaledH, true) // resize

        // center crop to size × size
        val x = ((scaledW - size) / 2).coerceAtLeast(0)
        val y = ((scaledH - size) / 2).coerceAtLeast(0)

        return Bitmap.createBitmap(resized, x, y, size, size)
    }

    /** Copy model file from assets → internal storage */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4096) // copy buffer
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

    /** Convert model output (NCHW) → Bitmap */
    private fun floatArrayToBitmap(
        floatArray: FloatArray,
        width: Int,
        height: Int,
        mean: FloatArray,
        std: FloatArray
    ): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)

        var idx = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                // extract R, G, B from NCHW layout
                val r = ((floatArray[idx] * std[0] + mean[0]) * 255).toInt().coerceIn(0, 255)
                val g = ((floatArray[idx + width * height] * std[1] + mean[1]) * 255).toInt().coerceIn(0, 255)
                val b = ((floatArray[idx + 2 * width * height] * std[2] + mean[2]) * 255).toInt().coerceIn(0, 255)

                // pack into ARGB pixel
                pixels[y * width + x] =
                    (0xFF shl 24) or (r shl 16) or (g shl 8) or b

                idx++
            }
        }

        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }
}