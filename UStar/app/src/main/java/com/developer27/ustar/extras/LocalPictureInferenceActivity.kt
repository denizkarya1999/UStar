package com.developer27.ustar.extras

import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
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
import com.developer27.ustar.machinelearning.Orientation_ResNet18
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

    // Model selector checkboxes
    private lateinit var checkCycleGAN: CheckBox
    private lateinit var checkOpticalRanging: CheckBox
    private lateinit var checkOrientation: CheckBox

    // Multiple bitmaps being processed
    private val selectedBitmaps = mutableListOf<Bitmap>()

    // Lazily-loaded model references
    private var opticalModel: Optical_Ranging_ResNet18? = null
    private var orientationModel: Orientation_ResNet18? = null

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

            var loadedCount = 0
            for (uri in uris) {
                val bmp = loadBitmapFromUri(uri)
                if (bmp != null) {
                    selectedBitmaps.add(bmp)
                    loadedCount++
                }
            }

            if (selectedBitmaps.isEmpty()) {
                Toast.makeText(this, "Failed to load selected images.", Toast.LENGTH_SHORT).show()
                imageOriginal.setImageDrawable(null)
                imageDenoised.setImageDrawable(null)
                btnRunInference.isEnabled = false
                return@registerForActivityResult
            }

            // Show the first selected image as a quick preview
            imageOriginal.setImageBitmap(selectedBitmaps.first())
            imageDenoised.setImageDrawable(null)

            textOpticalResult.text = "Optical Ranging Model —"
            textOrientationResult.text = "Orientation Guidance Model —"

            btnRunInference.isEnabled = true

            Toast.makeText(
                this,
                "Loaded $loadedCount image(s).",
                Toast.LENGTH_SHORT
            ).show()
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

        // Disable UI during inference
        progressBar.visibility = View.VISIBLE
        btnRunInference.isEnabled = false
        btnSelectImage.isEnabled = false

        lifecycleScope.launch {
            try {
                // -------------------------------------------------------------------------
                // 1) Optional CycleGAN Denoising (ONE RANDOM IMAGE)
                // -------------------------------------------------------------------------
                if (checkCycleGAN.isChecked) {
                    val randomIndex = Random.nextInt(selectedBitmaps.size)
                    val randomBitmap = selectedBitmaps[randomIndex]

                    val denoisedBitmap = withContext(Dispatchers.Default) {
                        Denoising_CycleGAN.load(this@LocalPictureInferenceActivity)
                        Denoising_CycleGAN.run(randomBitmap)
                    }

                    // Show random original + denoised pair
                    imageOriginal.setImageBitmap(randomBitmap)
                    imageDenoised.setImageBitmap(denoisedBitmap)
                } else {
                    // If CycleGAN not selected, just show first image as reference
                    imageOriginal.setImageBitmap(selectedBitmaps.first())
                    imageDenoised.setImageDrawable(null)
                }

                // -------------------------------------------------------------------------
                // 2) ResNet-18 Optical Ranging Model (ALL IMAGES)
                // -------------------------------------------------------------------------
                var opticalLastPrediction: String? = null
                var opticalCorrect = 0
                var opticalTotal = 0

                if (checkOpticalRanging.isChecked) {
                    withContext(Dispatchers.Default) {
                        if (opticalModel == null) {
                            opticalModel = Optical_Ranging_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }

                        selectedBitmaps.forEach { bmp ->
                            val result = opticalModel?.run(bmp)
                            if (result != null) {
                                opticalLastPrediction = result.topClass
                                opticalTotal++
                                if (gtOptical != null &&
                                    result.topClass.equals(gtOptical, ignoreCase = true)
                                ) {
                                    opticalCorrect++
                                }
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
                    } else {
                        textOpticalResult.text =
                            "Optical Ranging Model - Failed to load or run."
                    }
                } else {
                    textOpticalResult.text = "Optical Ranging Model —"
                }

                // -------------------------------------------------------------------------
                // 3) ResNet-18 Orientation Guidance Model (ALL IMAGES)
                // -------------------------------------------------------------------------
                var orientationLastPrediction: String? = null
                var orientationCorrect = 0
                var orientationTotal = 0

                if (checkOrientation.isChecked) {
                    withContext(Dispatchers.Default) {
                        if (orientationModel == null) {
                            orientationModel = Orientation_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }

                        selectedBitmaps.forEach { bmp ->
                            val result = orientationModel?.run(bmp)
                            if (result != null) {
                                orientationLastPrediction = result.topClass
                                orientationTotal++
                                if (gtOrientation != null &&
                                    result.topClass.equals(gtOrientation, ignoreCase = true)
                                ) {
                                    orientationCorrect++
                                }
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
                    } else {
                        textOrientationResult.text =
                            "Orientation Guidance Model - Failed to load or run."
                    }
                } else {
                    textOrientationResult.text = "Orientation Guidance Model —"
                }

            } catch (e: Exception) {
                // Any unhandled exception during inference
                Toast.makeText(
                    this@LocalPictureInferenceActivity,
                    "Inference failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            } finally {
                // Re-enable UI no matter what
                progressBar.visibility = View.GONE
                btnRunInference.isEnabled = true
                btnSelectImage.isEnabled = true
            }
        }
    }

    /**
     * Safely decode a Bitmap from gallery URI.
     */
    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            contentResolver.openInputStream(uri)?.use { input ->
                BitmapFactory.decodeStream(input)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}