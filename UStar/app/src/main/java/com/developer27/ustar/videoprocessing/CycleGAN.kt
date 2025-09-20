package com.developer27.ustar.videoprocessing

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

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

    /** Run inference on an image stored in assets/ and save the result into local storage. */
    fun runOnAssetToPictures(
        context: Context,
        assetImageName: String,
        outputFileName: String = "cyclegan_out.png",
        inputFileName: String = "cyclegan_in.png"
    ): Pair<File?, File?> {
        requireNotNull(module) { "Call load(context) before running inference." }

        // 1) Load asset → Bitmap
        val srcBmp = context.assets.open(assetImageName).use { input ->
            android.graphics.BitmapFactory.decodeStream(input)
                ?: throw IllegalArgumentException("Failed to decode asset: $assetImageName")
        }

        // 2) Run inference
        val outBmp = run(srcBmp)

        return if (Build.VERSION.SDK_INT >= 29) {
            val resolver = context.contentResolver

            // Delete existing with same names
            listOf(inputFileName, outputFileName).forEach { name ->
                val selection = "${MediaStore.Images.Media.DISPLAY_NAME}=?"
                resolver.delete(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, selection, arrayOf(name))
            }

            // Save input
            val inputValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, inputFileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/UStar")
            }
            resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, inputValues)?.let { uri ->
                resolver.openOutputStream(uri).use {
                    if (it != null) {
                        srcBmp.compress(Bitmap.CompressFormat.PNG, 100, it)
                    }
                }
            }

            // Save output
            val outValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, outputFileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/UStar")
            }
            resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, outValues)?.let { uri ->
                resolver.openOutputStream(uri).use {
                    if (it != null) {
                        outBmp.compress(Bitmap.CompressFormat.PNG, 100, it)
                    }
                }
            }

            Log.i("CycleGAN", "✅ Saved input as $inputFileName and output as $outputFileName in Pictures/UStar")

            val baseDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            Pair(File(baseDir, "UStar/$inputFileName"), File(baseDir, "UStar/$outputFileName"))
        } else {
            // Legacy write (pre-Android 10)
            val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
            val outDir = File(picturesDir, "UStar")
            if (!outDir.exists()) outDir.mkdirs()

            val inputFile = File(outDir, inputFileName)
            if (inputFile.exists()) inputFile.delete()
            FileOutputStream(inputFile).use { srcBmp.compress(Bitmap.CompressFormat.PNG, 100, it) }

            val outFile = File(outDir, outputFileName)
            if (outFile.exists()) outFile.delete()
            FileOutputStream(outFile).use { outBmp.compress(Bitmap.CompressFormat.PNG, 100, it) }

            Log.i("CycleGAN", "✅ Saved input as ${inputFile.absolutePath} and output as ${outFile.absolutePath}")
            Pair(inputFile, outFile)
        }
    }
}