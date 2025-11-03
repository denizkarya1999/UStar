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

    // Pass THIS view to our Renderer (not a generic Context)
    private val renderer = Renderer(this)

    private var prevX = 0f
    private var prevY = 0f
    private val rotationScale = 0.5f

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y
        if (event.action == MotionEvent.ACTION_MOVE) {
            val dx = x - prevX
            val dy = y - prevY
            renderer.rotateCube(dy * rotationScale, dx * rotationScale)
        }
        prevX = x
        prevY = y
        return true
    }

    /** Rebuild the cube texture from the latest file contents. */
    fun refresh() {
        queueEvent { renderer.loadTextureFromFile() }
    }

    /** Reads UStar_Cube_Prediction.txt and builds a 1024x1024 text bitmap. */
    internal fun buildTextBitmap(): Bitmap {
        val logFile = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS),
            "UStar_Cube_Prediction.txt"
        )

        val text = try {
            if (!logFile.exists()) {
                "[No Log]\nUStar 3D Cube\nOpenCV Initialized: false\nResNet-18 Prediction: none"
            } else {
                FileInputStream(logFile).use { it.bufferedReader().readText().trim() }
                    .ifEmpty { "[Empty File]\nUStar 3D Cube\nOpenCV Initialized: false\nResNet-18 Prediction: none" }
            }
        } catch (_: Exception) {
            "[Error Reading File]\nUStar 3D Cube\nOpenCV Initialized: false\nResNet-18 Prediction: none"
        }

        // inside ArMain.buildTextBitmap()
        val bmp = Bitmap.createBitmap(1024, 1024, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)

        // Light background so black text is readable (use white or light gray)
        canvas.drawColor(Color.WHITE)

        val title = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK            // ← black title
            textSize = 56f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        }
        val body = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK            // ← black body
            textSize = 44f
            typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
        }

        var y = 90f
        text.split("\n").forEachIndexed { idx, line ->
            if (idx == 1 && line.contains("UStar 3D Cube")) {
                canvas.drawText(line, 48f, y, title)
            } else {
                canvas.drawText(line, 48f, y, body)
            }
            y += 64f
        }
        return bmp
    }

    /** Creates an OpenGL texture from the given bitmap. */
    internal fun createTextureFromBitmap(bmp: Bitmap): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
        bmp.recycle()
        return textures[0]
    }

    /** Expose renderer if you need to tweak it externally. */
    fun getRenderer(): Renderer = renderer
}
