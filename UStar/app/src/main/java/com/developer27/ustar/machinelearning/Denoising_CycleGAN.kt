package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

/**
 * CycleGAN runner (PyTorch Mobile).
 * - Loads a scripted model from assets.
 * - Runs a single-image forward pass.
 * - Converts tensor output back to Bitmap.
 */
object Denoising_CycleGAN {
    private var module: Module? = null
    private const val IMG = 256 // model expects 256x256

    /** Load the scripted model from assets into app files dir (once). */
    fun load(context: Context, assetName: String = "UStar_Denoising_CycleGAN.ptl") {
        if (module == null) {
            val path = assetFilePath(context, assetName) // copy to files/ if needed
            module = Module.load(path) // load TorchScript module
            Log.i("Denoising_CycleGAN", "✅ Loaded CycleGAN model from $path")
        } else {
            Log.i("Denoising_CycleGAN", "ℹ️ Model already loaded, skipping reload.")
        }
    }

    /** Run stylization on a Bitmap and return the output Bitmap. */
    fun run(input: Bitmap): Bitmap {

        // Use the helper function to get a perfect 256x256 square
        val resized = centerCropResizeToSquare(input, 256)

        // Convert Bitmap -> Float32 tensor, normalize to [-1,1] via mean/std of 0.5
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0.5f, 0.5f, 0.5f),   // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)    // std
        )

        // Forward pass: IValue(tensor) -> output tensor
        val outTensor = module!!.forward(IValue.from(tensor)).toTensor()
        val outArray = outTensor.dataAsFloatArray // NCHW flattened floats

        // Convert tensor floats back to ARGB_8888 Bitmap
        return floatArrayToBitmap(
            outArray,
            IMG, IMG,
            floatArrayOf(0.5f, 0.5f, 0.5f),  // mean used to unnormalize
            floatArrayOf(0.5f, 0.5f, 0.5f)   // std used to unnormalize
        )
    }

    /** Resize while preserving aspect ratio, then center-crop to a square of (size x size). */
    private fun centerCropResizeToSquare(input: Bitmap, size: Int): Bitmap {
        val inW = input.width
        val inH = input.height

        // Match training: Resize shortest side to 1.12 * size (≈287 for size=256)
        val target = (size * 1.12f).toInt()

        val scale = if (inW < inH) {
            target.toFloat() / inW.toFloat()
        } else {
            target.toFloat() / inH.toFloat()
        }

        val scaledW = (inW * scale).toInt()
        val scaledH = (inH * scale).toInt()

        val scaled = Bitmap.createScaledBitmap(input, scaledW, scaledH, true)

        // Then center-crop to 256×256
        val cropX = ((scaledW - size) / 2).coerceAtLeast(0)
        val cropY = ((scaledH - size) / 2).coerceAtLeast(0)

        return Bitmap.createBitmap(scaled, cropX, cropY, size, size)
    }

    /** Ensure asset is available as a readable file; returns absolute path. */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath
        // Copy asset to internal storage
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

    /**
     * Convert model output (float array, NCHW order) into a Bitmap.
     * Expects channels in [-1,1]; unnormalizes with mean/std then maps to [0,255].
     */
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
                // NCHW layout: R at idx, G at idx+H*W, B at idx+2*H*W
                val r = ((floatArray[idx] * std[0] + mean[0]) * 255.0f).toInt().coerceIn(0, 255)
                val g = ((floatArray[idx + width * height] * std[1] + mean[1]) * 255.0f).toInt().coerceIn(0, 255)
                val b = ((floatArray[idx + 2 * width * height] * std[2] + mean[2]) * 255.0f).toInt().coerceIn(0, 255)
                // Pack into ARGB (opaque)
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                idx++
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height)
        return bmp
    }
}
