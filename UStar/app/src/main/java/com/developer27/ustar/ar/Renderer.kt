package com.developer27.ustar.ar

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class Renderer(private val arMain: ArMain) : GLSurfaceView.Renderer {

    private var cube: Cube? = null
    private var textureId: Int = 0

    private val projMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val vpMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)

    private var rotX = 20f
    private var rotY = -30f

    // scale down the cube
    private val cubeScale = 0.45f

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        // Black background
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)

        Matrix.setLookAtM(viewMatrix, 0, 0f, 0f, 4f, 0f, 0f, 0f, 0f, 1f, 0f)

        // You can still create a texture even if the cube renders white
        val bmp = arMain.buildTextBitmap()
        textureId = arMain.createTextureFromBitmap(bmp)

        cube = Cube(textureId)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val ratio = width.toFloat() / height.toFloat()
        Matrix.perspectiveM(projMatrix, 0, 45f, ratio, 0.1f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        Matrix.setIdentityM(modelMatrix, 0)
        // Smaller cube
        Matrix.scaleM(modelMatrix, 0, cubeScale, cubeScale, cubeScale)
        // Rotation
        Matrix.rotateM(modelMatrix, 0, rotX, 1f, 0f, 0f)
        Matrix.rotateM(modelMatrix, 0, rotY, 0f, 1f, 0f)

        Matrix.multiplyMM(vpMatrix, 0, projMatrix, 0, viewMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, vpMatrix, 0, modelMatrix, 0)

        cube?.draw(mvpMatrix)
    }

    fun rotateCube(dx: Float, dy: Float) {
        rotX += dx
        rotY += dy
    }

    /** Re-read the log file and update the texture (kept for compatibility). */
    fun loadTextureFromFile() {
        val bmp = arMain.buildTextBitmap()
        textureId = arMain.createTextureFromBitmap(bmp)
        cube?.setTexture(textureId)
    }
}