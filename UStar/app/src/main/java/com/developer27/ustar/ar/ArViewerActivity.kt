// ArViewerActivity.kt
package com.developer27.ustar.ar

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class ArViewerActivity : AppCompatActivity() {
    private lateinit var arMain: ArMain

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arMain = ArMain(this)
        setContentView(arMain)
    }

    override fun onResume() {
        super.onResume()
        // Re-read the latest UStar_Cube_Prediction.txt if needed
        arMain.refresh()
    }
}
