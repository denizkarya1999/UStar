package com.developer27.ustar.extras

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.developer27.ustar.R
import com.developer27.ustar.machinelearning.MiniDynaSpaPreprocessor

class LocalPhotoInferenceDynaSpaActivity : AppCompatActivity() {

    private lateinit var imageOriginal: ImageView
    private lateinit var imageMask: ImageView
    private lateinit var selectButton: Button
    private lateinit var runButton: Button
    private lateinit var progressBar: ProgressBar

    private var selectedBitmap: Bitmap? = null

    // Adjust this to control how much of the image is kept
    private val maskRate = 0.06f

    // Pick one image from gallery
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bitmap = uriToBitmap(it)
                if (bitmap != null) {
                    selectedBitmap = bitmap
                    imageOriginal.setImageBitmap(bitmap)
                    imageMask.setImageDrawable(null)
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
        selectButton = findViewById(R.id.btnSelectImage)
        runButton = findViewById(R.id.btnRunInference)
        progressBar = findViewById(R.id.inferenceProgress)

        // Loads UStar_MiniDynaSpa_Denoising.pt
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

            progressBar.visibility = View.VISIBLE
            runButton.isEnabled = false
            selectButton.isEnabled = false

            Thread {
                try {
                    val result = MiniDynaSpaPreprocessor.run(this, bitmap)

                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        runButton.isEnabled = true
                        selectButton.isEnabled = true

                        if (result == null) {
                            Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
                            return@runOnUiThread
                        }

                        val maskedBitmap = MiniDynaSpaPreprocessor.applyHardMaskToOriginal(
                            original = bitmap,
                            maskTensor = result.mask,
                            maskRate = maskRate
                        )

                        imageMask.setImageBitmap(maskedBitmap)

                        Toast.makeText(this, "DynaSpa inference complete", Toast.LENGTH_SHORT).show()
                    }

                } catch (e: Exception) {
                    runOnUiThread {
                        progressBar.visibility = View.GONE
                        runButton.isEnabled = true
                        selectButton.isEnabled = true
                        Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }.start()
        }
    }

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