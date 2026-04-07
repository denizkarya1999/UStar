package com.developer27.ustar.machinelearning.DynaSpa

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

object DynaSpaMaskProcessor {

    /**
     * Extracts the region defined by [boundingBox] from [source],
     * processes it with OpenCV (placeholder), and returns a new Bitmap
     * where the processed region is placed back into its original position
     * on a black background.
     */
    fun processBoundingBox(
        source: Bitmap,
        boundingBox: Rect
    ): Bitmap {
        // 1. Create a black background bitmap of the same size
        val resultBitmap = Bitmap.createBitmap(source.width, source.height, source.config)
        resultBitmap.eraseColor(Color.BLACK)

        // 2. Extract the sub-bitmap from the bounding box
        // Ensure the rect is within bounds
        val safeRect = Rect(
            boundingBox.left.coerceIn(0, source.width),
            boundingBox.top.coerceIn(0, source.height),
            boundingBox.right.coerceIn(0, source.width),
            boundingBox.bottom.coerceIn(0, source.height)
        )

        if (safeRect.width() <= 0 || safeRect.height() <= 0) {
            return resultBitmap
        }

        val croppedBitmap = Bitmap.createBitmap(
            source,
            safeRect.left,
            safeRect.top,
            safeRect.width(),
            safeRect.height()
        )

        // 3. Process with OpenCV (Placeholder)
        val processedCrop = mockOpenCVProcess(croppedBitmap)

        // 4. Draw the processed crop back onto the result bitmap
        val canvas = Canvas(resultBitmap)
        canvas.drawBitmap(processedCrop, safeRect.left.toFloat(), safeRect.top.toFloat(), null)

        return resultBitmap
    }

    // TODO: Implement OpenCV-based Enhanced DynaSpa Processing here.
    private fun mockOpenCVProcess(input: Bitmap): Bitmap {
        // Create a mutable copy of the input bitmap
        val result = input.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }

        // Draw a red border and a cross (X) as a test visual
        canvas.drawRect(0f, 0f, result.width.toFloat(), result.height.toFloat(), paint)
        canvas.drawLine(0f, 0f, result.width.toFloat(), result.height.toFloat(), paint)
        canvas.drawLine(result.width.toFloat(), 0f, 0f, result.height.toFloat(), paint)

        return result
    }
}
