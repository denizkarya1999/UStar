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

/**
 * Activity allowing the user to select a picture
 * and run multiple deep learning models:
 *  - CycleGAN-based denoising model
 *  - ResNet-18-based Optical Ranging model
 *  - ResNet-18-based Orientation Guidance model
 *
 * The user can choose which models to run with checkboxes.
 */
class LocalPictureInferenceActivity : AppCompatActivity() {

    // UI elements for displaying input + output images
    private lateinit var imageOriginal: ImageView
    private lateinit var imageDenoised: ImageView

    // UI text elements to show model predictions
    private lateinit var textOpticalResult: TextView
    private lateinit var textOrientationResult: TextView

    // Loading indicator while inference is running
    private lateinit var progressBar: ProgressBar

    // Action buttons
    private lateinit var btnSelectImage: AppCompatButton
    private lateinit var btnRunInference: AppCompatButton

    // Model selector checkboxes
    private lateinit var checkCycleGAN: CheckBox
    private lateinit var checkOpticalRanging: CheckBox
    private lateinit var checkOrientation: CheckBox

    // Bitmap being processed
    private var selectedBitmap: Bitmap? = null

    // Lazily-loaded model references
    private var opticalModel: Optical_Ranging_ResNet18? = null
    private var orientationModel: Orientation_ResNet18? = null

    /**
     * ActivityResultLauncher to open gallery picker
     * when user selects "Choose Image".
     */
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                // Load selected image
                loadBitmapFromUri(uri)?.let { bmp ->
                    selectedBitmap = bmp

                    // Update UI
                    imageOriginal.setImageBitmap(bmp)
                    imageDenoised.setImageDrawable(null)

                    textOpticalResult.text = "Optical Ranging Model: —"
                    textOrientationResult.text = "Orientation Guidance Model: —"

                    btnRunInference.isEnabled = true
                } ?: run {
                    Toast.makeText(this, "Failed to load image.", Toast.LENGTH_SHORT).show()
                }
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

        checkCycleGAN = findViewById(R.id.checkCycleGAN)
        checkOpticalRanging = findViewById(R.id.checkOpticalRanging)
        checkOrientation = findViewById(R.id.checkOrientation)

        // Button: Choose image from gallery
        btnSelectImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
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
        val bmp = selectedBitmap
        if (bmp == null) {
            Toast.makeText(this, "Please choose an image first.", Toast.LENGTH_SHORT).show()
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

        // Disable UI during inference
        progressBar.visibility = View.VISIBLE
        btnRunInference.isEnabled = false
        btnSelectImage.isEnabled = false

        lifecycleScope.launch {
            try {
                // -------------------------------------------------------------------------
                // 1) Optional CycleGAN Denoising
                // -------------------------------------------------------------------------
                val denoisedBitmap = withContext(Dispatchers.Default) {
                    if (checkCycleGAN.isChecked) {
                        // Load model (only loads once)
                        Denoising_CycleGAN.load(this@LocalPictureInferenceActivity)
                        Denoising_CycleGAN.run(bmp)
                    } else {
                        bmp
                    }
                }

                // Update denoised image preview if CycleGAN was selected
                if (checkCycleGAN.isChecked) {
                    imageDenoised.setImageBitmap(denoisedBitmap)
                } else {
                    imageDenoised.setImageDrawable(null)
                }

                // -------------------------------------------------------------------------
                // 2) ResNet-18 Optical Ranging Model (uses noised bitmap)
                // -------------------------------------------------------------------------
                if (checkOpticalRanging.isChecked) {
                    val opticalResult = withContext(Dispatchers.Default) {
                        // Lazy model loader
                        if (opticalModel == null) {
                            opticalModel = Optical_Ranging_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }
                        opticalModel?.run(bmp)
                    }

                    if (opticalResult != null) {
                        textOpticalResult.text = "Optical Ranging Model — " + opticalResult.topClass
                    } else {
                        textOpticalResult.text =
                            "Optical Ranging Model - Failed to load or run."
                    }
                } else {
                    textOpticalResult.text = "Optical Ranging Model —"
                }

                // -------------------------------------------------------------------------
                // 3) ResNet-18 Orientation Guidance Model (uses noised bitmap)
                // -------------------------------------------------------------------------
                if (checkOrientation.isChecked) {
                    val orientationResult = withContext(Dispatchers.Default) {
                        // Lazy model loader
                        if (orientationModel == null) {
                            orientationModel = Orientation_ResNet18.loadModel(
                                this@LocalPictureInferenceActivity
                            )
                        }
                        orientationModel?.run(bmp)
                    }

                    if (orientationResult != null) {
                        textOrientationResult.text = "Orientation Guidance Model - " + orientationResult.topClass
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