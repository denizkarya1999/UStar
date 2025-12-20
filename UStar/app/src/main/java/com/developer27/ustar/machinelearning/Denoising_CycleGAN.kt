package com.developer27.ustar.machinelearning

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

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
        // Resize shortest side to 1.12×256 and center-crop to 256×256 (bicubic)
        val pre = centerCropResizeToSquareBicubicOpenCV(input, 256, 1.12f)

        // Convert Bitmap → Float tensor and normalize to [-1, 1]
        val tensor = TensorImageUtils.bitmapToFloat32Tensor(
            pre,
            floatArrayOf(0.5f, 0.5f, 0.5f),   // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)    // std
        )

        // Run CycleGAN forward pass
        val outTensor = module!!.forward(IValue.from(tensor)).toTensor()

        // Extract NCHW float output
        val outArray = outTensor.dataAsFloatArray

        // Unnormalize and convert output tensor back to Bitmap
        return floatArrayToBitmap(
            outArray,
            IMG, IMG,
            floatArrayOf(0.5f, 0.5f, 0.5f),   // mean
            floatArrayOf(0.5f, 0.5f, 0.5f)    // std
        )
    }

    /** Matches torchvision: Resize(short side -> 1.12*size) with bicubic, then center-crop to size x size. */
    private fun centerCropResizeToSquareBicubicOpenCV(input: Bitmap, size: Int, scaleMult: Float = 1.12f): Bitmap {
        val inW = input.width
        val inH = input.height
        val target = (size * scaleMult).toInt() // 287 if size=256

        // scale so shortest side becomes target
        val scale = if (inW < inH) target.toFloat() / inW.toFloat() else target.toFloat() / inH.toFloat()
        val scaledW = (inW * scale).toInt().coerceAtLeast(size)
        val scaledH = (inH * scale).toInt().coerceAtLeast(size)

        // Bitmap -> Mat (RGBA)
        val srcRgba = Mat()
        Utils.bitmapToMat(input, srcRgba)

        // Convert RGBA -> RGB (torchvision works on RGB)
        val srcRgb = Mat()
        Imgproc.cvtColor(srcRgba, srcRgb, Imgproc.COLOR_RGBA2RGB)

        // Bicubic resize (matches PIL BICUBIC / torchvision)
        val resized = Mat()
        Imgproc.resize(srcRgb, resized, Size(scaledW.toDouble(), scaledH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)

        // Center crop to 256x256
        val cropX = ((scaledW - size) / 2).coerceAtLeast(0)
        val cropY = ((scaledH - size) / 2).coerceAtLeast(0)
        val roi = Rect(cropX, cropY, size, size)
        val cropped = Mat(resized, roi)

        // Mat (RGB) -> Bitmap (ARGB_8888)
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val croppedRgba = Mat()
        Imgproc.cvtColor(cropped, croppedRgba, Imgproc.COLOR_RGB2RGBA)
        Utils.matToBitmap(croppedRgba, out)

        // release mats
        srcRgba.release()
        srcRgb.release()
        resized.release()
        croppedRgba.release()

        return out
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
