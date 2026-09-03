package com.developer27.ustar.extras

import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatButton
import androidx.lifecycle.lifecycleScope
import com.developer27.ustar.R
import com.developer27.ustar.machinelearning.Denoising_CycleGAN
import com.developer27.ustar.machinelearning.Optical_Ranging_ResNet18
import com.developer27.ustar.machinelearning.Orientation_Guidance_ResNet18
import com.developer27.ustar.storage.PredictionLogWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random

/**
 * Activity allowing the user to select pictures
 * and run multiple deep learning models:
 *  - CycleGAN-based denoising model
 *  - ResNet-18-based Optical Ranging model
 *  - ResNet-18-based Orientation Guidance model
 *
 * Supports selecting MULTIPLE images:
 *  - CycleGAN shows ONE RANDOM image (original + denoised)
 *  - Optical & Orientation run on ALL selected images
 *  - User can enter ground truth labels to compute accuracy.
 */
class LocalPictureInferenceActivity : AppCompatActivity() {

    // UI elements for displaying input + output images
    private lateinit var imageOriginal: ImageView
    private lateinit var imageDenoised: ImageView

    // UI text elements to show model predictions
    private lateinit var textOpticalResult: TextView
    private lateinit var textOrientationResult: TextView

    // Ground-truth input for accuracy calculation
    private lateinit var editOpticalLabel: EditText
    private lateinit var editOrientationLabel: EditText

    // Loading indicator while inference is running
    private lateinit var progressBar: ProgressBar

    // Action buttons
    private lateinit var btnSelectImage: AppCompatButton
    private lateinit var btnRunInference: AppCompatButton
    private lateinit var btnLaunchAr: AppCompatButton   // NEW: Launch AR button

    // Model selector checkboxes
    private lateinit var checkCycleGAN: CheckBox
    private lateinit var checkOpticalRanging: CheckBox
    private lateinit var checkOrientation: CheckBox

    // Multiple bitmaps being processed
    private val selectedBitmaps = mutableListOf<Bitmap>()

    // Lazily-loaded model references
    private var opticalModel: Optical_Ranging_ResNet18? = null
    private var orientationModel: Orientation_Guidance_ResNet18? = null

    // --- Logging support (same style as VideoProcessor) ---
    private val logMessage = StringBuilder()

    /**
     * ActivityResultLauncher to open gallery picker
     * when user selects "Choose Images".
     *
     * Uses GetMultipleContents -> user can select multiple images.
     */
    private val pickImagesLauncher =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris: List<Uri>? ->
            selectedBitmaps.clear()

            if (uris.isNullOrEmpty()) {
                Toast.makeText(this, "No images selected.", Toast.LENGTH_SHORT).show()
                imageOriginal.setImageDrawable(null)
                imageDenoised.setImageDrawable(null)
                btnRunInference.isEnabled = false
                return@registerForActivityResult
            }

            progressBar.visibility = View.VISIBLE
            btnSelectImage.isEnabled = false
            btnRunInference.isEnabled = false

            lifecycleScope.launch {
                val loaded = withContext(Dispatchers.IO) {
                    uris.mapNotNull(::loadBitmapFromUri)
                }
                selectedBitmaps.addAll(loaded)

                progressBar.visibility = View.GONE
                btnSelectImage.isEnabled = true

                if (selectedBitmaps.isEmpty()) {
                    Toast.makeText(
                        this@LocalPictureInferenceActivity,
                        "Failed to load selected images.",
                        Toast.LENGTH_SHORT
                    ).show()
                    imageOriginal.setImageDrawable(null)
                    imageDenoised.setImageDrawable(null)
                    return@launch
                }

                imageOriginal.setImageBitmap(selectedBitmaps.first())
                imageDenoised.setImageDrawable(null)
                textOpticalResult.text = "Optical Ranging Model —"
                textOrientationResult.text = "Orientation Guidance Model —"
                btnRunInference.isEnabled = true

                Toast.makeText(
                    this@LocalPictureInferenceActivity,
                    "Loaded ${loaded.size} image(s).",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Lock orientation + keep screen awake for model inference
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_local_picture_inference)

        // --- UI Binding ---
        imageOriginal = findViewById(R.id.imageOriginal)
        imageDenoised = findViewById(R.id.imageDenoised)
        textOpticalResult = findViewById(R.id.textOpticalResult)
        textOrientationResult = findViewById(R.id.textOrientationResult)
        progressBar = findViewById(R.id.inferenceProgress)

        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnRunInference = findViewById(R.id.btnRunInference)
        btnLaunchAr = findViewById(R.id.btnLaunchAr)      // NEW

        checkCycleGAN = findViewById(R.id.checkCycleGAN)
        checkOpticalRanging = findViewById(R.id.checkOpticalRanging)
        checkOrientation = findViewById(R.id.checkOrientation)

        // NEW: ground-truth label inputs
        editOpticalLabel = findViewById(R.id.editOpticalLabel)
        editOrientationLabel = findViewById(R.id.editOrientationLabel)

        // Button: Choose images from gallery (multiple)
        btnSelectImage.setOnClickListener {
            pickImagesLauncher.launch("image/*")
        }

        // Button: Run selected models
        btnRunInference.setOnClickListener {
            runSelectedModels()
        }

        // Button: Launch AR Activity
        btnLaunchAr.setOnClickListener {
            try {
                val intent = Intent(
                    this,
                    com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java
                )
                startActivity(intent)
            } catch (e: Exception) {
                Toast.makeText(
                    this,
                    "Unable to launch AR Activity: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    }

    /**
     * Run only the models selected by the user via checkboxes.
     * Uses Kotlin coroutines to ensure:
     *  - UI thread remains responsive
     *  - Heavy ML inference runs on background threads
     */
    private fun runSelectedModels() {
        if (selectedBitmaps.isEmpty()) {
            Toast.makeText(this, "Please choose at least one image first.", Toast.LENGTH_SHORT)
                .show()
            return
        }

        // Require at least one model selected
        if (!checkCycleGAN.isChecked &&
            !checkOpticalRanging.isChecked &&
            !checkOrientation.isChecked
        ) {
            Toast.makeText(this, "Select at least one model.", Toast.LENGTH_SHORT).show()
            return
        }

        // Get ground-truth labels (optional)
        val gtOptical = editOpticalLabel.text.toString().trim().ifEmpty { null }
        val gtOrientation = editOrientationLabel.text.toString().trim().ifEmpty { null }

        // --- Initialize log for this run ---
        // IMPORTANT: do NOT add extra header/date here.
        // VideoProcessor's writeLogToFile already adds:
        // "UStar UIOD Tag Features" + "Prediction Date: ..."
        // So here we only log details and predictions.
        logMessage.clear()
        logMessage.appendLine("LocalPictureInference: totalImages=${selectedBitmaps.size}")
        logMessage.appendLine("GT Optical: ${gtOptical ?: "N/A"}")
        logMessage.appendLine("GT Orientation: ${gtOrientation ?: "N/A"}")
        logMessage.appendLine(
            "Models: " +
                    "CycleGAN=${checkCycleGAN.isChecked}, " +
                    "Optical=${checkOpticalRanging.isChecked}, " +
                    "Orientation=${checkOrientation.isChecked}"
        )
        logMessage.appendLine("--------------------------------------------------")

        // Disable UI during inference
        progressBar.visibility = View.VISIBLE
        btnRunInference.isEnabled = false
        btnSelectImage.isEnabled = false
        btnLaunchAr.isEnabled = false

        lifecycleScope.launch {
            // We'll keep these outside so we can write final
            // Distance | Orientation line at the end.
            var opticalLastPrediction: String? = null
            var orientationLastPrediction: String? = null

            try {
                // -------------------------------------------------------------------------
                // 1) Optional CycleGAN Denoising (ONE RANDOM IMAGE)
                // -------------------------------------------------------------------------
                if (checkCycleGAN.isChecked) {
                    val randomIndex = Random.nextInt(selectedBitmaps.size)
                    val randomBitmap = selectedBitmaps[randomIndex]

                    val startTime = System.currentTimeMillis()
                    val denoisedBitmap = withContext(Dispatchers.Default) {
                        Denoising_CycleGAN.load(this@LocalPictureInferenceActivity)
                        Denoising_CycleGAN.run(randomBitmap)
                    }
                    val endTime = System.currentTimeMillis()

                    // Show random original + denoised pair
                    imageOriginal.setImageBitmap(randomBitmap)
                    imageDenoised.setImageBitmap(denoisedBitmap)

                    logMessage.appendLine("CycleGAN: Denoised sample index = $randomIndex")
                    logMessage.appendLine("CycleGAN Inference Time: ${endTime - startTime} ms")
                } else {
                    // If CycleGAN not selected, just show first image as reference
                    imageOriginal.setImageBitmap(selectedBitmaps.first())
                    imageDenoised.setImageDrawable(null)

                    logMessage.appendLine("CycleGAN: not run in this session.")
                }

                // -------------------------------------------------------------------------
                // 2) ResNet-18 Optical Ranging Model (ALL IMAGES)
                // -------------------------------------------------------------------------
                var opticalCorrect = 0
                var opticalTotal = 0

                if (checkOpticalRanging.isChecked) {
                    var totalTime = 0L
                    withContext(Dispatchers.Default) {
                        if (opticalModel == null) {
                            opticalModel = Optical_Ranging_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }

                        selectedBitmaps.forEachIndexed { index, bmp ->
                            val startTime = System.currentTimeMillis()
                            val result = opticalModel?.run(bmp)
                            val endTime = System.currentTimeMillis()
                            
                            if (result != null) {
                                opticalLastPrediction = result.topClass
                                opticalTotal++
                                val inferenceTime = endTime - startTime
                                totalTime += inferenceTime

                                // Log per-image prediction
                                logMessage.appendLine(
                                    "OpticalRanging: image#$index => pred=${result.topClass}, " +
                                            "gt=${gtOptical ?: "N/A"}, time=$inferenceTime ms"
                                )

                                if (gtOptical != null &&
                                    result.topClass.equals(gtOptical, ignoreCase = true)
                                ) {
                                    opticalCorrect++
                                }
                            } else {
                                logMessage.appendLine("OpticalRanging: image#$index => NULL result")
                            }
                        }
                    }

                    if (opticalTotal > 0) {
                        val last = opticalLastPrediction ?: "N/A"
                        val accText = if (gtOptical != null) {
                            "  |  Correct: $opticalCorrect / $opticalTotal"
                        } else {
                            ""
                        }
                        textOpticalResult.text =
                            "Optical Ranging Model — Last: $last$accText"

                        logMessage.appendLine(
                            "OpticalRanging Summary: last=$opticalLastPrediction, " +
                                    "correct=$opticalCorrect, total=$opticalTotal, avgTime=${totalTime / opticalTotal} ms"
                        )
                        logMessage.appendLine("Total Optical Ranging Inference Time: $totalTime ms")
                    } else {
                        textOpticalResult.text =
                            "Optical Ranging Model - Failed to load or run."
                        logMessage.appendLine("OpticalRanging: No images processed (load/run failed).")
                    }
                } else {
                    textOpticalResult.text = "Optical Ranging Model —"
                    logMessage.appendLine("OpticalRanging: not run in this session.")
                }

                // -------------------------------------------------------------------------
                // 3) ResNet-18 Orientation Guidance Model (ALL IMAGES)
                // -------------------------------------------------------------------------
                var orientationCorrect = 0
                var orientationTotal = 0

                if (checkOrientation.isChecked) {
                    var totalTime = 0L
                    withContext(Dispatchers.Default) {
                        if (orientationModel == null) {
                            orientationModel = Orientation_Guidance_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }

                        selectedBitmaps.forEachIndexed { index, bmp ->
                            val startTime = System.currentTimeMillis()
                            val result = orientationModel?.run(bmp)
                            val endTime = System.currentTimeMillis()
                            
                            if (result != null) {
                                orientationLastPrediction = result.topClass
                                orientationTotal++
                                val inferenceTime = endTime - startTime
                                totalTime += inferenceTime

                                // Log per-image prediction
                                logMessage.appendLine(
                                    "Orientation: image#$index => pred=${result.topClass}, " +
                                            "gt=${gtOrientation ?: "N/A"}, time=$inferenceTime ms"
                                )

                                if (gtOrientation != null &&
                                    result.topClass.equals(gtOrientation, ignoreCase = true)
                                ) {
                                    orientationCorrect++
                                }
                            } else {
                                logMessage.appendLine("Orientation: image#$index => NULL result")
                            }
                        }
                    }

                    if (orientationTotal > 0) {
                        val last = orientationLastPrediction ?: "N/A"
                        val accText = if (gtOrientation != null) {
                            "  |  Correct: $orientationCorrect / $orientationTotal"
                        } else {
                            ""
                        }
                        textOrientationResult.text =
                            "Orientation Guidance Model — Last: $last$accText"

                        logMessage.appendLine(
                            "Orientation Summary: last=$orientationLastPrediction, " +
                                    "correct=$orientationCorrect, total=$orientationTotal, avgTime=${totalTime / orientationTotal} ms"
                        )
                        logMessage.appendLine("Total Orientation Inference Time: $totalTime ms")
                    } else {
                        textOrientationResult.text =
                            "Orientation Guidance Model - Failed to load or run."
                        logMessage.appendLine("Orientation: No images processed (load/run failed).")
                    }
                } else {
                    textOrientationResult.text = "Orientation Guidance Model —"
                    logMessage.appendLine("Orientation: not run in this session.")
                }

                logMessage.appendLine("--------------------------------------------------")

                // 🔴 KEY PART: add final combined line exactly like VideoProcessor
                val distanceForLog = opticalLastPrediction ?: "N/A"
                val orientationForLog = orientationLastPrediction ?: "N/A"
                val combinedPrediction =
                    "Distance: $distanceForLog | Orientation: $orientationForLog"
                logMessage.appendLine(combinedPrediction)

                // -------------------------------------------------------------------------
                // 4) Write log to file (same style as VideoProcessor)
                // -------------------------------------------------------------------------
                withContext(Dispatchers.IO) {
                    writeLogToFile()
                }

            } catch (e: Exception) {
                logMessage.appendLine("ERROR during inference: ${e.message}")

                // Try to write partial log as well
                withContext(Dispatchers.IO) {
                    writeLogToFile()
                }
            } finally {
                // Re-enable UI no matter what
                progressBar.visibility = View.GONE
                btnRunInference.isEnabled = true
                btnSelectImage.isEnabled = true
                btnLaunchAr.isEnabled = true
            }
        }
    }

    /**
     * Safely decode a Bitmap from gallery URI.
     */
    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { input ->
                BitmapFactory.decodeStream(input, null, bounds)
            }
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

            var sampleSize = 1
            while (bounds.outWidth / sampleSize > MAX_IMAGE_DIMENSION ||
                bounds.outHeight / sampleSize > MAX_IMAGE_DIMENSION
            ) {
                sampleSize *= 2
            }

            val options = BitmapFactory.Options().apply { inSampleSize = sampleSize }
            contentResolver.openInputStream(uri)?.use { input ->
                BitmapFactory.decodeStream(input, null, options)
            }
        } catch (e: Exception) {
            Log.e("LocalPictureInference", "Unable to decode selected image", e)
            null
        }
    }

    /**
     * Writes the full log with date and header to Documents/UStar_Cube_Prediction.txt
     * (same filename / style as your VideoProcessor logger).
     *
     * NOTE: On Android 11+ you may need to handle scoped storage properly
     * or adjust this path.
     */
    private fun writeLogToFile() {
        PredictionLogWriter.write(this, logMessage)
    }

    private companion object {
        const val MAX_IMAGE_DIMENSION = 2048
    }
}
