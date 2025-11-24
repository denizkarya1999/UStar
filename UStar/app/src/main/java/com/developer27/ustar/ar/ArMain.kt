package com.developer27.ustar.ar

import android.content.Context
import android.graphics.*
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.os.Environment
import android.util.AttributeSet
import android.view.MotionEvent
import java.io.File
import java.io.FileInputStream

class ArMain @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    // Pass THIS view to our Renderer
    private val renderer = Renderer(this)

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    /**
     * Do NOT let the user change the arrow orientation.
     * We simply consume the touch and ignore it.
     */
    override fun onTouchEvent(event: MotionEvent): Boolean {
        // If you want the view to remain scrollable by parent, return false.
        // If you want it to "eat" touches, return true.
        return true
    }

    /** Ask renderer to reload texture (arrow) from txt file. */
    fun refresh() {
        queueEvent { renderer.loadTextureFromFile() }
    }

    /** Map orientation string to a rotation angle in degrees. */
    internal fun orientationToAngle(ori: String): Float {
        val o = ori.trim().lowercase()
        return when (o) {
            "north"      ->   0f
            "northeast"  ->  45f
            "east"       ->  90f
            "southeast"  -> 135f
            "south"      -> 180f
            "southwest"  -> 225f
            "west"       -> 270f
            "northwest"  -> 315f
            else         ->   0f   // fallback
        }
    }

    /**
     * Reads UStar_Cube_Prediction.txt and builds a 1024x1024 bitmap
     * containing a compass arrow (rotated by orientation) and distance label
     * (1m–4m) written on the arrow.
     */
    internal fun buildArrowBitmap(): Bitmap {
        val logFile = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
            "UStar_Cube_Prediction.txt"
        )

        val rawText = try {
            if (!logFile.exists()) {
                "UStar UIOD Tag Features\n" +
                        "Prediction Date: N/A\n" +
                        "OpenCV Initialization Status: false\n" +
                        "Distance: 1M | Orientation: North"
            } else {
                FileInputStream(logFile).use { it.bufferedReader().readText().trim() }
                    .ifEmpty {
                        "UStar UIOD Tag Features\n" +
                                "Prediction Date: N/A\n" +
                                "OpenCV Initialization Status: false\n" +
                                "Distance: 1M | Orientation: North"
                    }
            }
        } catch (_: Exception) {
            "UStar UIOD Tag Features\n" +
                    "Prediction Date: N/A\n" +
                    "OpenCV Initialization Status: false\n" +
                    "Distance: 1M | Orientation: North"
        }

        // --- Parse distance (1–4m) and orientation from the file ---
        var distanceMeters: Int? = null
        var orientationLabel = "North"

        rawText.lines().forEach { line ->
            if (line.startsWith("Distance:", ignoreCase = true)) {
                val parts = line.split("|")

                if (parts.isNotEmpty()) {
                    val distRegex = Regex("""Distance:\s*(\d+)\s*[Mm]""")
                    val match = distRegex.find(parts[0])
                    distanceMeters = match?.groupValues?.get(1)?.toIntOrNull()
                }

                if (parts.size > 1) {
                    val oriRegex = Regex("""Orientation:\s*([A-Za-z]+)""")
                    val matchOri = oriRegex.find(parts[1])
                    orientationLabel = matchOri?.groupValues?.get(1) ?: orientationLabel
                }
            }
        }

        val clampedDist = distanceMeters?.coerceIn(1, 4) ?: 1
        val distanceText = "${clampedDist}m"
        val angle = orientationToAngle(orientationLabel)

        val bmp = Bitmap.createBitmap(1024, 1024, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)

        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val arrowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }

        val arrowOutline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 6f
        }

        // Centered, smaller debug text
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 40f                // smaller so it fits nicely
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
            textAlign = Paint.Align.CENTER  // <-- key for centering
        }

        val centerX = bmp.width / 2f

        // Centered lines at the top
        canvas.drawText(
            "ORIENTATION: ${orientationLabel.uppercase()}",
            centerX,
            80f,
            labelPaint
        )
        canvas.drawText(
            "DISTANCE: ${distanceText}",
            centerX,
            140f,
            labelPaint
        )

        // --- Arrow path as before ---
        val arrowPath = Path().apply {
            moveTo(0f, -280f)
            lineTo(-90f, -130f)
            lineTo(-40f, -130f)
            lineTo(-40f, 260f)
            lineTo(40f, 260f)
            lineTo(40f, -130f)
            lineTo(90f, -130f)
            close()
        }

        canvas.save()
        canvas.translate(512f, 512f)
        canvas.rotate(angle)

        canvas.drawPath(arrowPath, arrowPaint)
        canvas.drawPath(arrowPath, arrowOutline)

        canvas.restore()

        return bmp
    }

    /** Creates an OpenGL texture from the given bitmap. */
    internal fun createTextureFromBitmap(bmp: Bitmap): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0])
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MIN_FILTER,
            GLES20.GL_LINEAR
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MAG_FILTER,
            GLES20.GL_LINEAR
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        bmp.recycle()
        return textures[0]
    }

    /** Expose renderer if needed. */
    fun getRenderer(): Renderer = renderer
}