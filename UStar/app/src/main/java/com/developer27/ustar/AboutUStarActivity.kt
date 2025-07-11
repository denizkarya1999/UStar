package com.developer27.ustar

import android.content.Intent
import android.content.pm.ActivityInfo
import android.net.Uri
import android.os.Bundle
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import com.developer27.ustar.databinding.ActivityAboutXameraBinding

class AboutUStarActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAboutXameraBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Disable screen rotation (lock to portrait)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        // Prevent screen from sleeping
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        binding = ActivityAboutXameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // When the title container is clicked, open the URL in a browser.
        binding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        // When the UM logo is clicked, show the website.
        binding.umLogo.setOnClickListener {
            val url = "https://umich.edu/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }
    }
}