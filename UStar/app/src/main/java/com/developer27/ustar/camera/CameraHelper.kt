package com.developer27.ustar.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Range
import android.util.Size
import android.util.SparseIntArray
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresPermission
import com.developer27.ustar.MainActivity
import com.developer27.ustar.databinding.ActivityMainBinding
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import kotlin.math.max

public class CameraHelper(
    public val activity: MainActivity,
    public val viewBinding: ActivityMainBinding
) {

    // ------------------------------------------------------------------------
    // Public fields
    // ------------------------------------------------------------------------
    public val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    public var cameraDevice: CameraDevice? = null
    public var cameraCaptureSession: CameraCaptureSession? = null
    public var captureRequestBuilder: CaptureRequest.Builder? = null
    public var imageReader: ImageReader? = null        // persistent for still capture
    public var previewSize: Size? = null
    public var videoSize: Size? = null
    public var sensorArraySize: Rect? = null
    public var backgroundThread: HandlerThread? = null
    public var backgroundHandler: Handler? = null
    public var zoomLevel: Float = 1.0f
    public val maxZoom: Float = 10.0f
    public var isFrontCamera: Boolean = false

    // ------------------------------------------------------------------------
    // Companion (public)
    // ------------------------------------------------------------------------
    public companion object {
        public val ORIENTATIONS: SparseIntArray = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    // ------------------------------------------------------------------------
    // State callback (public)
    // ------------------------------------------------------------------------
    public val stateCallback: CameraDevice.StateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreview()
        }
        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            cameraDevice = null
        }
        override fun onError(camera: CameraDevice, error: Int) {
            camera.close()
            cameraDevice = null
            activity.finish()
        }
    }

    // ------------------------------------------------------------------------
    // Thread helpers
    // ------------------------------------------------------------------------
    public fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    public fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        backgroundThread?.join()
        backgroundThread = null
        backgroundHandler = null
    }

    // ------------------------------------------------------------------------
    // Open / Close camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    public fun openCamera() {
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
        previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
        videoSize   = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

        cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
    }

    public fun closeCamera() {
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
    }

    // ------------------------------------------------------------------------
    // Preview Creation
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    public fun createCameraPreview() {
        val device = cameraDevice ?: return
        val texture = viewBinding.viewFinder.surfaceTexture ?: return
        val size = previewSize ?: return

        texture.setDefaultBufferSize(size.width, size.height)
        val previewSurface = Surface(texture)

        // Persistent JPEG reader
        imageReader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2)
        val readerSurface = imageReader!!.surface

        captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewSurface)
            applyRollingShutter15Hz()
            applyZoom()
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)
        }

        device.createCaptureSession(
            listOf(previewSurface, readerSurface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    cameraCaptureSession = session
                    updatePreview()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Toast.makeText(activity, "Preview config failed.", Toast.LENGTH_SHORT).show()
                }
            },
            backgroundHandler
        )
    }

    public fun updatePreview() {
        val session = cameraCaptureSession ?: return
        val builder = captureRequestBuilder ?: return
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        builder.set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)
        session.setRepeatingRequest(builder.build(), null, backgroundHandler)
    }

    // ------------------------------------------------------------------------
    // Still capture
    // ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Still‑capture that restarts preview *and* writes to "Pictures/UStar Pictures"
// ------------------------------------------------------------------------
    public fun takePhoto(onPhotoSaved: ((file: File) -> Unit)? = null) {
        val device   = cameraDevice        ?: return
        val session  = cameraCaptureSession ?: return
        val reader   = imageReader         ?: return

        // --- build still request ---
        val captureBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)

            // ----‑‑‑ your zoom / exposure code remains unchanged  ‑‑‑‑‑‑-
            sensorArraySize?.let { rect ->
                if (zoomLevel > 1.0f) {
                    val ratio = 1 / zoomLevel
                    val w = (rect.width() * ratio).toInt()
                    val h = (rect.height() * ratio).toInt()
                    val l = (rect.width() - w) / 2
                    val t = (rect.height() - h) / 2
                    set(CaptureRequest.SCALER_CROP_REGION, Rect(l, t, l + w, t + h))
                }
            }
            // 15 Hz / AWB / manual‑exposure logic (same as before) …
            val chara = cameraManager.getCameraCharacteristics(getCameraId())
            val caps  = chara.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            val manual = caps?.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(15, 15))
            if (manual) {
                val expRange = chara.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                val isoRange = chara.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                if (expRange != null && isoRange != null) {
                    val ns = (1_000_000_000L / 15).coerceIn(expRange.lower, expRange.upper)
                    set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                    set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.SENSOR_EXPOSURE_TIME, ns)
                    set(CaptureRequest.SENSOR_SENSITIVITY, max(isoRange.lower, 100))
                } else {
                    set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                }
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            }
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)
        }

        // --- save JPEG when ready ---
        reader.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage()
            val buffer = img.planes[0].buffer
            val bytes  = ByteArray(buffer.remaining()).also { buffer.get(it) }
            img.close()

            // ==> Pictures/UStar Pictures/<timestamp>.jpg
            val picturesDir = android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_PICTURES)
            val ustarDir = java.io.File(picturesDir, "Exported Pictures from UStar")
            if (!ustarDir.exists()) ustarDir.mkdirs()
            val file = java.io.File(ustarDir, "UStar_${System.currentTimeMillis()}.jpg")
            java.io.FileOutputStream(file).use { it.write(bytes) }

            activity.runOnUiThread {
                android.widget.Toast.makeText(
                    activity,
                    "Photo saved: ${file.absolutePath}",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
                onPhotoSaved?.invoke(file)
            }
        }, backgroundHandler)

        // --- capture then resume preview ---
        session.stopRepeating()
        session.capture(
            captureBuilder.build(),
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    sess: CameraCaptureSession,
                    req: CaptureRequest,
                    result: TotalCaptureResult
                ) { resumePreview() }
                override fun onCaptureFailed(
                    sess: CameraCaptureSession,
                    req: CaptureRequest,
                    failure: CaptureFailure
                ) { resumePreview() }   // still resume if it fails
            },
            backgroundHandler
        )
    }

    /** Re‑submit the stored repeating preview request so the camera keeps running. */
    private fun resumePreview() {
        try {
            val sess = cameraCaptureSession ?: return
            val builder = captureRequestBuilder ?: return
            sess.setRepeatingRequest(builder.build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Rolling shutter (preview)
    // ------------------------------------------------------------------------
    public fun applyRollingShutter15Hz() {
        captureRequestBuilder?.let { builder ->
            val chara = cameraManager.getCameraCharacteristics(getCameraId())
            val caps  = chara.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            val manual = caps?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(15, 15))
            if (!manual) {
                builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            }
        }
    }

    // ------------------------------------------------------------------------
    // Zoom controls (preview)
    // ------------------------------------------------------------------------
    public fun setupZoomControls() {
        val handler = Handler(activity.mainLooper)
        var zoomInRun : Runnable? = null
        var zoomOutRun: Runnable? = null

        viewBinding.zoomInButton.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRun = object : Runnable {
                        override fun run() {
                            zoomLevel = (zoomLevel + 0.1f).coerceAtMost(maxZoom)
                            applyZoom()
                            handler.postDelayed(this, 50)
                        }
                    }
                    handler.post(zoomInRun!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    handler.removeCallbacks(zoomInRun!!)
                    true
                }
                else -> false
            }
        }

        viewBinding.zoomOutButton.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRun = object : Runnable {
                        override fun run() {
                            zoomLevel = (zoomLevel - 0.1f).coerceAtLeast(1.0f)
                            applyZoom()
                            handler.postDelayed(this, 50)
                        }
                    }
                    handler.post(zoomOutRun!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    handler.removeCallbacks(zoomOutRun!!)
                    true
                }
                else -> false
            }
        }
    }

    public fun applyZoom() {
        val builder = captureRequestBuilder ?: return
        val rect    = sensorArraySize ?: return
        val ratio = 1 / zoomLevel
        val w = (rect.width() * ratio).toInt()
        val h = (rect.height() * ratio).toInt()
        val l = (rect.width() - w) / 2
        val t = (rect.height() - h) / 2
        builder.set(CaptureRequest.SCALER_CROP_REGION, Rect(l, t, l + w, t + h))
        cameraCaptureSession?.setRepeatingRequest(builder.build(), null, backgroundHandler)
    }

    // ------------------------------------------------------------------------
    // Camera‑ID + size helpers
    // ------------------------------------------------------------------------
    public fun getCameraId(): String {
        cameraManager.cameraIdList.forEach { id ->
            val facing = cameraManager
                .getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (!isFrontCamera && facing == CameraCharacteristics.LENS_FACING_BACK) return id
            if ( isFrontCamera && facing == CameraCharacteristics.LENS_FACING_FRONT) return id
        }
        return cameraManager.cameraIdList.first()
    }

    public fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetW = 1280
        val targetH = 720
        choices.find { it.width == targetW && it.height == targetH }?.let { return it }
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }
}
