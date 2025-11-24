package com.developer27.ustar.ar

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class Renderer(private val arMain: ArMain) : GLSurfaceView.Renderer {

    private var arrow: Arrow? = null
    private var textureId: Int = 0

    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val vpMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)

    // Large, but still fits fully in view
    private val arrowScale = 1.4f   // instead of 1.8f

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        // Black background
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)

        // Camera looking at origin
        Matrix.setLookAtM(
            viewMatrix, 0,
            0f, 0f, 4f,   // eye
            0f, 0f, 0f,   // center
            0f, 1f, 0f    // up
        )

        // Build initial arrow texture from file
        val bmp = arMain.buildArrowBitmap()
        textureId = arMain.createTextureFromBitmap(bmp)

        arrow = Arrow(textureId)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val ratio = width.toFloat() / height.toFloat()

        // Simple perspective projection
        Matrix.perspectiveM(
            projMatrix, 0,
            45f, ratio,
            0.1f, 100f
        )
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        Matrix.setIdentityM(modelMatrix, 0)
        // Scale quad
        Matrix.scaleM(modelMatrix, 0, arrowScale, arrowScale, arrowScale)

        Matrix.multiplyMM(vpMatrix, 0, projMatrix, 0, viewMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, vpMatrix, 0, modelMatrix, 0)

        arrow?.draw(mvpMatrix)
    }

    fun loadTextureFromFile() {
        val bmp = arMain.buildArrowBitmap()
        textureId = arMain.createTextureFromBitmap(bmp)
        arrow?.setTexture(textureId)
    }
}