package com.developer27.ustar.extras

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.developer27.ustar.R
import com.developer27.ustar.machinelearning.MiniDynaSpaPreprocessor

class LocalPhotoInferenceDynaSpaActivity : AppCompatActivity() {

    private lateinit var imageOriginal: ImageView
    private lateinit var imageMask: ImageView
    private lateinit var predictionLabel: TextView
    private lateinit var selectButton: Button
    private lateinit var runButton: Button
    private lateinit var progressBar: ProgressBar

    private var selectedBitmap: Bitmap? = null

    // Pick image from gallery
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bitmap = uriToBitmap(it)
                if (bitmap != null) {
                    selectedBitmap = bitmap
                    imageOriginal.setImageBitmap(bitmap)
                    imageMask.setImageDrawable(null)
                    predictionLabel.text = "Prediction: No result yet"
                } else {
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_local_dynaspa_inference)

        imageOriginal = findViewById(R.id.imageOriginal)
        imageMask = findViewById(R.id.imageMask)
        predictionLabel = findViewById(R.id.predictionLabel)
        selectButton = findViewById(R.id.btnSelectImage)
        runButton = findViewById(R.id.btnRunInference)
        progressBar = findViewById(R.id.inferenceProgress)

        // Load model once
        MiniDynaSpaPreprocessor.load(this)

        selectButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        runButton.setOnClickListener {
            val bitmap = selectedBitmap
            if (bitmap == null) {
                Toast.makeText(this, "Select an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Update UI while inference is running
            progressBar.visibility = View.VISIBLE
            runButton.isEnabled = false
            selectButton.isEnabled = false
            predictionLabel.text = "Prediction: Running..."

            Thread {
                try {
                    val result = MiniDynaSpaPreprocessor.run(this, bitmap)

                    runOnUiThread {
                        // Restore UI
                        progressBar.visibility = View.GONE
                        runButton.isEnabled = true
                        selectButton.isEnabled = true

                        if (result == null) {
                            predictionLabel.text = "Prediction: Inference failed"
                            Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                            return@runOnUiThread
                        }

                        // Show feature map
                        val featureMapBitmap =
                            MiniDynaSpaPreprocessor.featureMapToHeatmapBitmap(
                                featureMapTensor = result.featureMap,
                                targetWidth = bitmap.width,
                                targetHeight = bitmap.height
                            )

                        imageMask.setImageBitmap(featureMapBitmap)

                        // Store probabilities
                        val probs = result.probabilities

                        // Ensure 2 classes
                        val prob0 = if (probs.size > 0) probs[0] else 0f
                        val prob1 = if (probs.size > 1) probs[1] else 0f

                        // Show final prediction text
                        predictionLabel.text =
                            "No UOID tag: ${String.format("%.3f", prob0)}\n" +
                                    "UOID tag present: ${String.format("%.3f", prob1)}"
                    }

                } catch (e: Exception) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        runButton.isEnabled = true
                        selectButton.isEnabled = true
                        predictionLabel.text = "Prediction: Error"
                        Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }
    }

    // Convert URI to bitmap
    private fun uriToBitmap(uri: Uri): Bitmap? {
        return try {
            contentResolver.openInputStream(uri)?.use { input ->
                BitmapFactory.decodeStream(input)
            }
        } catch (e: Exception) {
            null
        }
    }
}