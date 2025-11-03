package com.developer27.ustar.ar

import android.opengl.GLES20
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class Cube(private var textureId: Int) {

    private var program = 0
    private var aPos = 0
    private var aTex = 0
    private var uMVP = 0
    private var uTex0 = 0

    private val vertexBuffer: FloatBuffer
    private val uvBuffer: FloatBuffer
    private val vertexCount: Int

    init {
        val positions = floatArrayOf(
            // +X
            1f,-1f,-1f,  1f, 1f,-1f,  1f, 1f, 1f,
            1f,-1f,-1f,  1f, 1f, 1f,  1f,-1f, 1f,
            // -X
            -1f,-1f, 1f, -1f, 1f, 1f, -1f, 1f,-1f,
            -1f,-1f, 1f, -1f, 1f,-1f, -1f,-1f,-1f,
            // +Y
            -1f, 1f,-1f, -1f, 1f, 1f,  1f, 1f, 1f,
            -1f, 1f,-1f,  1f, 1f, 1f,  1f, 1f,-1f,
            // -Y
            -1f,-1f, 1f, -1f,-1f,-1f,  1f,-1f,-1f,
            -1f,-1f, 1f,  1f,-1f,-1f,  1f,-1f, 1f,
            // +Z
            -1f,-1f, 1f,  1f,-1f, 1f,  1f, 1f, 1f,
            -1f,-1f, 1f,  1f, 1f, 1f, -1f, 1f, 1f,
            // -Z
            1f,-1f,-1f, -1f,-1f,-1f, -1f, 1f,-1f,
            1f,-1f,-1f, -1f, 1f,-1f,  1f, 1f,-1f
        )

        val faceUV = floatArrayOf(
            0f,1f, 1f,1f, 1f,0f,
            0f,1f, 1f,0f, 0f,0f
        )
        val uvs = FloatArray(faceUV.size * 6) { i -> faceUV[i % faceUV.size] }

        vertexBuffer = positions.toBuffer()
        uvBuffer = uvs.toBuffer()
        vertexCount = positions.size / 3

        // ⬇️ Use textured shader (NOT solid white)
        program = linkProgram(VS, FS_TEXTURED)
        aPos = GLES20.glGetAttribLocation(program, "aPosition")
        aTex = GLES20.glGetAttribLocation(program, "aTexCoord")
        uMVP = GLES20.glGetUniformLocation(program, "uMVP")
        uTex0 = GLES20.glGetUniformLocation(program, "uTex0")

        GLES20.glUseProgram(program)
        GLES20.glUniform1i(uTex0, 0) // texture unit 0
    }

    fun setTexture(newTextureId: Int) {
        textureId = newTextureId
    }

    fun draw(mvp: FloatArray) {
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(uMVP, 1, false, mvp, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)

        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 3 * 4, vertexBuffer)

        GLES20.glEnableVertexAttribArray(aTex)
        GLES20.glVertexAttribPointer(aTex, 2, GLES20.GL_FLOAT, false, 0, uvBuffer)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, vertexCount)

        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aTex)
    }

    private fun FloatArray.toBuffer(): FloatBuffer =
        ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(this@toBuffer)
            position(0)
        }

    private fun linkProgram(vsSrc: String, fsSrc: String): Int {
        fun compile(type: Int, src: String): Int {
            val id = GLES20.glCreateShader(type)
            GLES20.glShaderSource(id, src)
            GLES20.glCompileShader(id)
            val ok = IntArray(1)
            GLES20.glGetShaderiv(id, GLES20.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) throw RuntimeException("Shader compile error: " + GLES20.glGetShaderInfoLog(id))
            return id
        }
        val vs = compile(GLES20.GL_VERTEX_SHADER, VS)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fsSrc)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vs)
        GLES20.glAttachShader(prog, fs)
        GLES20.glBindAttribLocation(prog, 0, "aPosition")
        GLES20.glBindAttribLocation(prog, 1, "aTexCoord")
        GLES20.glLinkProgram(prog)
        val ok = IntArray(1)
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, ok, 0)
        if (ok[0] == 0) throw RuntimeException("Program link error: " + GLES20.glGetProgramInfoLog(prog))
        return prog
    }

    companion object {
        private const val VS = """
            attribute vec3 aPosition;
            attribute vec2 aTexCoord;
            uniform mat4 uMVP;
            varying vec2 vTex;
            void main() {
                vTex = aTexCoord;
                gl_Position = uMVP * vec4(aPosition, 1.0);
            }
        """

        // ⬇️ Textured shader to show the bitmap (white background, black letters)
        private const val FS_TEXTURED = """
            precision mediump float;
            varying vec2 vTex;
            uniform sampler2D uTex0;
            void main() {
                gl_FragColor = texture2D(uTex0, vTex);
            }
        """
    }
}