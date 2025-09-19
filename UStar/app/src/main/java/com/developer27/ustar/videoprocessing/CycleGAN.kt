package com.developer27.ustar.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.FileOutputStream
import java.io.File

object CycleGAN {
    private var module: Module? = null
    private const val IMG = 256

    fun load(context: Context, assetName: String = "G_AB.ptl") {
        if (module == null) {
            val path = assetFilePath(context, assetName)
            module = Module.load(path)
            Log.i("CycleGAN", "✅ Loaded CycleGAN model from $path")
        } else {
            Log.i("CycleGAN", "ℹ️ Model already loaded, skipping reload.")
        }
    }

    fun run(input: Bitmap): Bitmap {
        val resized = Bitmap.createScaledBitmap(input, IMG, IMG, true)

        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0.5f, 0.5f, 0.5f),   // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)    // std
        )

        val outTensor = module!!.forward(IValue.from(tensor)).toTensor()
        val outArray = outTensor.dataAsFloatArray

        val outBitmap = floatArrayToBitmap(
            outArray,
            IMG, IMG,
            floatArrayOf(0.5f, 0.5f, 0.5f),  // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)   // std
        )
        return outBitmap
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (true) {
                    read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }
        return file.absolutePath
    }

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
                // CycleGAN output is NCHW: 3 × H × W
                val r = ((floatArray[idx] * std[0] + mean[0]) * 255.0f).toInt().coerceIn(0, 255)
                val g = ((floatArray[idx + width * height] * std[1] + mean[1]) * 255.0f).toInt().coerceIn(0, 255)
                val b = ((floatArray[idx + 2 * width * height] * std[2] + mean[2]) * 255.0f).toInt().coerceIn(0, 255)
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                idx++
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }
}