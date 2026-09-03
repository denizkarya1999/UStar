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
import kotlin.math.abs
import kotlin.math.max

/**
 * CameraHelper
 * ------------
 * Thin wrapper around Camera2 for this app's needs:
 *  - Discovers front/back camera IDs and preferred output sizes.
 *  - Opens/closes the camera and maintains a repeating preview request.
 *  - Builds a persistent JPEG ImageReader for still capture.
 *  - Applies zoom via SCALER_CROP_REGION and a 15 Hz target frame rate for preview.
 *  - Provides long-press zoom-in / zoom-out UI bindings.
 *
 * Lifecycle overview:
 *  - [startBackgroundThread] / [stopBackgroundThread] manage a HandlerThread for camera callbacks.
 *  - [openCamera] checks sizes and calls CameraManager.openCamera(...).
 *  - [stateCallback] receives the Device and calls [createCameraPreview].
 *  - [createCameraPreview] configures the session with Preview + JPEG ImageReader surfaces.
 *  - [updatePreview] submits the repeating preview request.
 *  - [takePhoto] issues a still capture, writes JPEG to public Pictures, then resumes preview.
 *
 * Notes:
 *  - Orientation mapping is provided via [ORIENTATIONS] but JPEG orientation is not explicitly set.
 *  - External public storage writes may require special handling on modern Android (scoped storage).
 */
public class CameraHelper(
    public val activity: MainActivity,
    public val viewBinding: ActivityMainBinding
) {

    // ------------------------------------------------------------------------
    // Public fields (shared state / components)
    // ------------------------------------------------------------------------

    /** Entry point for Camera2 system services. */
    public val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** The opened camera device (null until [stateCallback.onOpened]). */
    public var cameraDevice: CameraDevice? = null

    /** The active capture session that owns the repeating preview request. */
    public var cameraCaptureSession: CameraCaptureSession? = null

    /** Builder for preview requests; reused to keep controls (zoom, AWB, etc.) consistent. */
    public var captureRequestBuilder: CaptureRequest.Builder? = null

    /** JPEG reader kept alive for still captures. */
    public var imageReader: ImageReader? = null

    /** Chosen sizes for preview and (potential) video. */
    public var previewSize: Size? = null
    public var videoSize: Size? = null
    public var photoSize: Size? = null

    /** Full active sensor area; used to compute SCALER_CROP_REGION for zoom. */
    public var sensorArraySize: Rect? = null

    /** Background thread + handler for Camera2 work (session callbacks, image saving, etc.). */
    public var backgroundThread: HandlerThread? = null
    public var backgroundHandler: Handler? = null

    /** Current zoom factor (1.0 = no zoom). Adjusted by UI. */
    public var zoomLevel: Float = 1.0f

    /** Camera-supported cap for zoomLevel. */
    public var maxZoom: Float = 1.0f
        private set

    /** Which lens is active; toggled from UI. */
    public var isFrontCamera: Boolean = false

    @Volatile
    private var cameraRequested: Boolean = false

    // ------------------------------------------------------------------------
    // Companion (public)
    // ------------------------------------------------------------------------

    /**
     * Display rotation → JPEG rotation degrees mapping (as a utility reference).
     * Not applied directly in this class for still capture; add JPEG_ORIENTATION if needed.
     */
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

    /**
     * CameraDevice lifecycle callback:
     *  - onOpened: store device and start preview session
     *  - onDisconnected/onError: close and null out device; finish activity on error
     */
    public val stateCallback: CameraDevice.StateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            if (!cameraRequested) {
                camera.close()
                return
            }
            cameraDevice = camera
            createCameraPreview()
        }
        override fun onDisconnected(camera: CameraDevice) {
            cameraRequested = false
            camera.close()
            cameraDevice = null
        }
        override fun onError(camera: CameraDevice, error: Int) {
            cameraRequested = false
            camera.close()
            cameraDevice = null
            activity.runOnUiThread {
                Toast.makeText(activity, "Unable to open camera (error $error).", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ------------------------------------------------------------------------
    // Thread helpers
    // ------------------------------------------------------------------------

    /** Start a dedicated background thread/handler to receive Camera2 callbacks and do disk I/O. */
    public fun startBackgroundThread() {
        if (backgroundThread?.isAlive == true) return
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    /** Stop and join the background thread; null out handler references. */
    public fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        backgroundThread?.join()
        backgroundThread = null
        backgroundHandler = null
    }

    // ------------------------------------------------------------------------
    // Open / Close camera
    // ------------------------------------------------------------------------

    /**
     * Discover sizes and open the selected camera.
     * Requires CAMERA permission (validated by caller).
     */
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    public fun openCamera() {
        if (cameraDevice != null || cameraRequested) return
        val callbackHandler = backgroundHandler ?: return
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        maxZoom = (characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
            ?: 1.0f).coerceAtLeast(1.0f)
        zoomLevel = zoomLevel.coerceIn(1.0f, maxZoom)

        // Pick sizes for preview and potential video streams
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
        previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
        videoSize   = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))
        photoSize = chooseOptimalSize(map.getOutputSizes(ImageFormat.JPEG))

        // Asynchronous open; result delivered to [stateCallback]
        cameraRequested = true
        try {
            cameraManager.openCamera(cameraId, stateCallback, callbackHandler)
        } catch (e: Exception) {
            cameraRequested = false
            throw e
        }
    }

    /** Close session + device; safe to call from lifecycle callbacks. */
    public fun closeCamera() {
        cameraRequested = false
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
        captureRequestBuilder = null
    }

    // ------------------------------------------------------------------------
    // Preview Creation
    // ------------------------------------------------------------------------

    /**
     * Build a preview session with:
     *  - Preview Surface (TextureView)
     *  - Persistent JPEG ImageReader (for still capture)
     *
     * Sets initial request controls: 15 Hz rolling shutter target, zoom, AWB, color correction.
     */
    @SuppressLint("MissingPermission")
    public fun createCameraPreview() {
        val device = cameraDevice ?: return
        val texture = viewBinding.viewFinder.surfaceTexture ?: return
        val size = previewSize ?: return
        val stillSize = photoSize ?: size

        // Back the TextureView with a buffer of the chosen preview size
        texture.setDefaultBufferSize(size.width, size.height)
        val previewSurface = Surface(texture)

        // Persistent JPEG reader (maxImages=2 allows one queued image while saving)
        imageReader?.close()
        imageReader = ImageReader.newInstance(stillSize.width, stillSize.height, ImageFormat.JPEG, 2)
        val readerSurface = imageReader!!.surface

        // Build a PREVIEW request and set default controls
        captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewSurface)
            applyRollingShutter15Hz(this) // AE target fps; manual exposure is handled in still capture
            applyZoom(this)               // apply current zoom crop region
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)
        }

        // Create a session that outputs to both preview and ImageReader (for still capture)
        device.createCaptureSession(
            listOf(previewSurface, readerSurface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    cameraCaptureSession = session
                    updatePreview()
                }
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    activity.runOnUiThread {
                        Toast.makeText(activity, "Preview config failed.", Toast.LENGTH_SHORT).show()
                    }
                }
            },
            backgroundHandler
        )
    }

    /** Submit/refresh the repeating preview request with current controls. */
    public fun updatePreview() {
        val session = cameraCaptureSession ?: return
        val builder = captureRequestBuilder ?: return
        builder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        builder.set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)
        try {
            session.setRepeatingRequest(builder.build(), null, backgroundHandler)
        } catch (e: Exception) {
            android.util.Log.e("CameraHelper", "Unable to start camera preview", e)
        }
    }

    // ------------------------------------------------------------------------
    // Still capture
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Still‑capture that restarts preview *and* writes to "Pictures/UStar Pictures"
    // ------------------------------------------------------------------------
    /**
     * Capture a single JPEG using the persistent ImageReader and save it to
     * `Pictures/Exported Pictures from UStar/UStar_<timestamp>.jpg`.
     *
     * Flow:
     *  - Build a TEMPLATE_STILL_CAPTURE request.
     *  - Apply current zoom via SCALER_CROP_REGION.
     *  - Try to use manual exposure (if supported), else fall back to AUTO AE.
     *  - Listen for the JPEG in ImageReader and write it to disk on background thread.
     *  - Resume the repeating preview after capture completes (or even if it fails).
     *
     * @param onPhotoSaved optional callback invoked on the UI thread with the saved File.
     */
    public fun takePhoto(onPhotoSaved: ((file: File) -> Unit)? = null) {
        val device   = cameraDevice        ?: return
        val session  = cameraCaptureSession ?: return
        val reader   = imageReader         ?: return

        // --- build still request ---
        val captureBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)

            // Re-apply zoom for the still capture (crop at sensor level)
            sensorArraySize?.let { rect ->
                if (zoomLevel > 1.0f) {
                    val ratio = 1 / zoomLevel
                    val w = (rect.width() * ratio).toInt()
                    val h = (rect.height() * ratio).toInt()
                    val l = rect.left + (rect.width() - w) / 2
                    val t = rect.top + (rect.height() - h) / 2
                    set(CaptureRequest.SCALER_CROP_REGION, Rect(l, t, l + w, t + h))
                }
            }

            // Target ~15 fps; if manual sensor control exists, set exposure and ISO; else rely on AE
            val chara = cameraManager.getCameraCharacteristics(getCameraId())
            val caps  = chara.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            val manual = caps?.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
            chooseTargetFpsRange(chara)?.let {
                set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
            }
            if (manual) {
                val expRange = chara.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
                val isoRange = chara.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
                if (expRange != null && isoRange != null) {
                    // 1/15s in nanoseconds, clamped to device-supported range
                    val ns = (1_000_000_000L / 15).coerceIn(expRange.lower, expRange.upper)
                    set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                    set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                    set(CaptureRequest.SENSOR_EXPOSURE_TIME, ns)
                    set(CaptureRequest.SENSOR_SENSITIVITY, max(isoRange.lower, 100))
                } else {
                    // If ranges are missing, revert to automatic exposure
                    set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                    set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                }
            } else {
                // No full manual capability: stick with AE
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            }

            // White balance & color correction
            set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
            set(CaptureRequest.COLOR_CORRECTION_MODE, CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY)

            // (Optional) Consider setting JPEG_ORIENTATION here using [ORIENTATIONS] + display rotation.
        }

        // --- save JPEG when ready ---
        // For every capture the listener is (re)assigned; last assignment wins, which is fine for single-shot flow.
        reader.setOnImageAvailableListener({ r ->
            val img = r.acquireNextImage()
            val buffer = img.planes[0].buffer
            val bytes  = ByteArray(buffer.remaining()).also { buffer.get(it) }
            img.close()

            // Save to Pictures/Exported Pictures from UStar
            val picturesDir = android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_PICTURES)
            val ustarDir = java.io.File(picturesDir, "Exported Pictures from UStar")
            if (!ustarDir.exists()) ustarDir.mkdirs()
            val file = java.io.File(ustarDir, "UStar_${System.currentTimeMillis()}.jpg")
            java.io.FileOutputStream(file).use { it.write(bytes) }

            // Notify on UI thread
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
                ) { resumePreview() }   // resume even on failure to keep preview alive
            },
            backgroundHandler
        )
    }

    /** Re-submit the stored repeating preview request so the camera keeps running after a still shot. */
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

    /**
     * Apply a 15 Hz target FPS range for preview.
     * If manual sensor controls are unsupported, we keep CONTROL/AE in AUTO.
     * (Manual exposure for preview is not forced here; we only enforce manual in still capture.)
     */
    public fun applyRollingShutter15Hz() {
        captureRequestBuilder?.let { builder -> applyRollingShutter15Hz(builder) }
    }

    private fun applyRollingShutter15Hz(builder: CaptureRequest.Builder) {
        val chara = cameraManager.getCameraCharacteristics(getCameraId())
        val caps = chara.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val manual = caps?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true
        chooseTargetFpsRange(chara)?.let {
            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
        }
        if (!manual) {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
        }
    }

    private fun chooseTargetFpsRange(characteristics: CameraCharacteristics): Range<Int>? {
        val ranges = characteristics.get(
            CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
        ) ?: return null
        return ranges.minWithOrNull(
            compareBy<Range<Int>>(
                { if (it.lower <= 15 && it.upper >= 15) 0 else 1 },
                { abs(it.lower - 15) + abs(it.upper - 15) },
                { it.upper - it.lower }
            )
        )
    }

    // ------------------------------------------------------------------------
    // Zoom controls (preview)
    // ------------------------------------------------------------------------

    /**
     * Bind long-press handlers to "zoom in/out" buttons:
     *  - While pressed: adjust zoomLevel every 50 ms and re-apply crop region.
     *  - Release/cancel: stop adjustments.
     */
    public fun setupZoomControls() {
        val handler = Handler(activity.mainLooper)
        var zoomInRun : Runnable? = null
        var zoomOutRun: Runnable? = null

        viewBinding.zoomInButton.setOnTouchListener { view, e ->
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
                    zoomInRun?.let(handler::removeCallbacks)
                    if (e.action == MotionEvent.ACTION_UP) view.performClick()
                    true
                }
                else -> false
            }
        }

        viewBinding.zoomOutButton.setOnTouchListener { view, e ->
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
                    zoomOutRun?.let(handler::removeCallbacks)
                    if (e.action == MotionEvent.ACTION_UP) view.performClick()
                    true
                }
                else -> false
            }
        }
    }

    /**
     * Compute and apply SCALER_CROP_REGION based on [zoomLevel], then update the repeating request.
     * The crop is centered within [sensorArraySize].
     */
    public fun applyZoom() {
        val builder = captureRequestBuilder ?: return
        applyZoom(builder)
        try {
            cameraCaptureSession?.setRepeatingRequest(builder.build(), null, backgroundHandler)
        } catch (e: Exception) {
            android.util.Log.e("CameraHelper", "Unable to apply camera zoom", e)
        }
    }

    private fun applyZoom(builder: CaptureRequest.Builder) {
        val rect    = sensorArraySize ?: return
        zoomLevel = zoomLevel.coerceIn(1.0f, maxZoom)
        val ratio = 1 / zoomLevel
        val w = (rect.width() * ratio).toInt()
        val h = (rect.height() * ratio).toInt()
        val l = rect.left + (rect.width() - w) / 2
        val t = rect.top + (rect.height() - h) / 2
        builder.set(CaptureRequest.SCALER_CROP_REGION, Rect(l, t, l + w, t + h))
    }

    // ------------------------------------------------------------------------
    // Camera‑ID + size helpers
    // ------------------------------------------------------------------------

    /**
     * Choose a camera ID based on [isFrontCamera] preference.
     * Falls back to the first ID if a specific facing isn't found.
     */
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

    /**
     * Pick a size, preferring 1280×720 if available, else the smallest area to reduce memory/bandwidth.
     * (You can change this policy to pick the largest under a cap, or aspect-ratio closest to your UI view.)
     */
    public fun chooseOptimalSize(choices: Array<Size>): Size {
        require(choices.isNotEmpty()) { "Camera reported no supported output sizes" }
        val targetW = 1280
        val targetH = 720
        choices.find { it.width == targetW && it.height == targetH }?.let { return it }
        return choices.minByOrNull { abs(it.width - targetW) + abs(it.height - targetH) }
            ?: choices[0]
    }
}
