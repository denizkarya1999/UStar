package com.developer27.ustar.extras

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.developer27.ustar.R
import com.developer27.ustar.machinelearning.DynaSpa.DynaSpaMaskProcessor
import com.developer27.ustar.machinelearning.DynaSpa.MiniDynaSpaPreprocessor
import com.developer27.ustar.storage.PredictionLogWriter
import java.util.Locale

class LocalPhotoInferenceDynaSpaActivity : AppCompatActivity() {

    private lateinit var imageOriginal: ImageView
    private lateinit var imageModelInput: ImageView
    private lateinit var imageMask: ImageView
    private lateinit var imageBoxMask: ImageView
    private lateinit var imageDynaSpa: ImageView
    private lateinit var predictionLabel: TextView
    private lateinit var selectButton: Button
    private lateinit var runButton: Button
    private lateinit var progressBar: ProgressBar

    private var selectedBitmap: Bitmap? = null

    // Global variable for combined log message
    private var logMessage: StringBuilder = StringBuilder()

    // Pick image from gallery
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bitmap = uriToBitmap(it)
                if (bitmap != null) {
                    selectedBitmap = bitmap
                    imageOriginal.setImageBitmap(bitmap)
                    imageModelInput.setImageDrawable(null)
                    imageMask.setImageDrawable(null)
                    imageBoxMask.setImageDrawable(null)
                    imageDynaSpa.setImageDrawable(null)
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
        imageModelInput = findViewById(R.id.imageModelInput)
        imageMask = findViewById(R.id.imageMask)
        imageBoxMask = findViewById(R.id.imageBoxMask)
        imageDynaSpa = findViewById(R.id.imageDynaSpa)
        predictionLabel = findViewById(R.id.predictionLabel)
        selectButton = findViewById(R.id.btnSelectImage)
        runButton = findViewById(R.id.btnRunInference)
        progressBar = findViewById(R.id.inferenceProgress)

        selectButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        runButton.setOnClickListener {
            val bitmap = selectedBitmap
            if (bitmap == null) {
                Toast.makeText(this, "Select an image first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Update UI during inference
            progressBar.visibility = View.VISIBLE
            runButton.isEnabled = false
            selectButton.isEnabled = false
            predictionLabel.text = "Prediction: Running..."

            Thread {
                try {
                    logMessage = StringBuilder()
                    val startTime = System.currentTimeMillis()
                    val result = MiniDynaSpaPreprocessor.run(this, bitmap)
                    val endTime = System.currentTimeMillis()
                    
                    if (result != null) {
                        logMessage.appendLine("DynaSpa Model Inference Time: ${endTime - startTime} ms")
                        
                        val box = MiniDynaSpaPreprocessor.extractBoundingBoxFromFeatureMap(
                            featureMapTensor = result.featureMap,
                            outputWidth = result.processedBitmap.width,
                            outputHeight = result.processedBitmap.height
                        )

                        val dynaSpaStartTime = System.currentTimeMillis()
                        val dynaSpaProcessedBitmap = if (MiniDynaSpaPreprocessor.shouldShowBoundingBox(result.predictedClass)) {
                            DynaSpaMaskProcessor.processBoundingBox(
                                source = result.processedBitmap,
                                boundingBox = box
                            )
                        } else {
                            Bitmap.createBitmap(
                                result.processedBitmap.width,
                                result.processedBitmap.height,
                                Bitmap.Config.ARGB_8888
                            ).apply { eraseColor(android.graphics.Color.BLACK) }
                        }
                        val dynaSpaEndTime = System.currentTimeMillis()
                        logMessage.appendLine("DynaSpa Reconstruction Time: ${dynaSpaEndTime - dynaSpaStartTime} ms")

                        val boxMaskedBitmap =
                            MiniDynaSpaPreprocessor.featureMapToBoundingBoxMaskedBitmap(
                                processedBitmap = result.processedBitmap,
                                featureMapTensor = result.featureMap,
                                predictedClass = result.predictedClass
                            )

                        val featureMapBitmap =
                            MiniDynaSpaPreprocessor.featureMapToHeatmapBitmap(
                                featureMapTensor = result.featureMap
                            )

                        val probs = result.probabilities
                        val prob0 = if (probs.isNotEmpty()) probs[0] else 0f
                        val prob1 = if (probs.size > 1) probs[1] else 0f
                        val predictionText =
                            "No UOID tag: ${String.format(Locale.getDefault(), "%.3f", prob0)}\n" +
                                "UOID tag present: ${String.format(Locale.getDefault(), "%.3f", prob1)}"
                        logMessage.appendLine(predictionText.replace("\n", " | "))
                        writeLogToFile()

                        runOnUiThread {
                            // Restore UI
                            progressBar.visibility = View.GONE
                            runButton.isEnabled = true
                            selectButton.isEnabled = true

                            // Show model input (the 1024 crop resized to 224)
                            imageModelInput.setImageBitmap(result.processedBitmap)

                            // Show feature-map heatmap
                            imageMask.setImageBitmap(featureMapBitmap)

                            // Show bounding-box masked cropped image
                            imageBoxMask.setImageBitmap(boxMaskedBitmap)

                            imageDynaSpa.setImageBitmap(dynaSpaProcessedBitmap)

                            predictionLabel.text = predictionText
                        }
                    } else {
                        runOnUiThread {
                            progressBar.visibility = View.GONE
                            runButton.isEnabled = true
                            selectButton.isEnabled = true
                            predictionLabel.text = "Prediction: Inference failed"
                            Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                        }
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
            Log.e("LocalDynaSpa", "Unable to decode selected image", e)
            null
        }
    }

    /** Writes the full log with date and header to Documents/UStar_Cube_Prediction.txt */
    private fun writeLogToFile() {
        PredictionLogWriter.write(this, logMessage)
    }

    private companion object {
        const val MAX_IMAGE_DIMENSION = 2048
    }
}
