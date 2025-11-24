package com.developer27.ustar.ar

import android.opengl.GLES20

class Arrow(private var textureId: Int) {

    // A simple square (two triangles) in X/Y plane, Z=0
    //   (-1,  1)        (1, 1)
    //      0-------------1
    //      |           / |
    //      |         /   |
    //      |       /     |
    //      |     /       |
    //      |   /         |
    //      | /           |
    //      3-------------2
    //   (-1, -1)        (1,-1)
    private val vertices = floatArrayOf(
        // X,   Y,   Z,   U,  V
        -1f,  1f,  0f,   0f, 0f,  // 0 top-left
        1f,  1f,  0f,   1f, 0f,  // 1 top-right
        1f, -1f,  0f,   1f, 1f,  // 2 bottom-right
        -1f, -1f,  0f,   0f, 1f   // 3 bottom-left
    )

    private val indices = shortArrayOf(
        0, 1, 2,
        0, 2, 3
    )

    private val vertexBuffer = GlUtils.createFloatBuffer(vertices)
    private val indexBuffer = GlUtils.createShortBuffer(indices)

    private val program: Int
    private val aPositionHandle: Int
    private val aTexCoordHandle: Int
    private val uMVPMatrixHandle: Int
    private val uTextureHandle: Int

    init {
        val vertexShaderCode = """
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            uniform mat4 uMVPMatrix;

            void main() {
                gl_Position = uMVPMatrix * aPosition;
                vTexCoord = aTexCoord;
            }
        """.trimIndent()

        val fragmentShaderCode = """
            precision mediump float;
            varying vec2 vTexCoord;
            uniform sampler2D uTexture;

            void main() {
                gl_FragColor = texture2D(uTexture, vTexCoord);
            }
        """.trimIndent()

        val vertexShader = GlUtils.loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = GlUtils.loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        program = GLES20.glCreateProgram().also { prog ->
            GLES20.glAttachShader(prog, vertexShader)
            GLES20.glAttachShader(prog, fragmentShader)
            GLES20.glLinkProgram(prog)
        }

        aPositionHandle = GLES20.glGetAttribLocation(program, "aPosition")
        aTexCoordHandle = GLES20.glGetAttribLocation(program, "aTexCoord")
        uMVPMatrixHandle = GLES20.glGetUniformLocation(program, "uMVPMatrix")
        uTextureHandle = GLES20.glGetUniformLocation(program, "uTexture")
    }

    fun setTexture(newTextureId: Int) {
        textureId = newTextureId
    }

    fun draw(mvpMatrix: FloatArray) {
        GLES20.glUseProgram(program)

        // Position
        vertexBuffer.position(0)
        GLES20.glEnableVertexAttribArray(aPositionHandle)
        GLES20.glVertexAttribPointer(
            aPositionHandle,
            3,               // x,y,z
            GLES20.GL_FLOAT,
            false,
            5 * 4,           // stride = (3 pos + 2 uv) * 4 bytes
            vertexBuffer
        )

        // TexCoord
        vertexBuffer.position(3)
        GLES20.glEnableVertexAttribArray(aTexCoordHandle)
        GLES20.glVertexAttribPointer(
            aTexCoordHandle,
            2,               // u,v
            GLES20.GL_FLOAT,
            false,
            5 * 4,
            vertexBuffer
        )

        // MVP
        GLES20.glUniformMatrix4fv(uMVPMatrixHandle, 1, false, mvpMatrix, 0)

        // Texture
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
        GLES20.glUniform1i(uTextureHandle, 0)

        // Draw
        GLES20.glDrawElements(
            GLES20.GL_TRIANGLES,
            indices.size,
            GLES20.GL_UNSIGNED_SHORT,
            indexBuffer
        )

        GLES20.glDisableVertexAttribArray(aPositionHandle)
        GLES20.glDisableVertexAttribArray(aTexCoordHandle)
    }
}

/**
 * Tiny GL utilities for buffers & shader compilation.
 */
object GlUtils {

    fun loadShader(type: Int, shaderCode: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, shaderCode)
        GLES20.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw RuntimeException("Could not compile shader $type: $log")
        }
        return shader
    }

    fun createFloatBuffer(data: FloatArray) =
        java.nio.ByteBuffer.allocateDirect(data.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(data)
                position(0)
            }

    fun createShortBuffer(data: ShortArray) =
        java.nio.ByteBuffer.allocateDirect(data.size * 2)
            .order(java.nio.ByteOrder.nativeOrder())
            .asShortBuffer()
            .apply {
                put(data)
                position(0)
            }
}
