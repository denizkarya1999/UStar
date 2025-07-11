package com.developer27.ustar

import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreference
import com.developer27.ustar.videoprocessing.Settings

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Lock to portrait & keep screen on
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.settings_activity)
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // ───── Bounding Box Preference ─────
            findPreference<SwitchPreference>("enable_bounding_box")
                ?.setOnPreferenceChangeListener { _, newValue ->
                    val enabled = newValue as Boolean
                    Settings.BoundingBox.enableBoundingBox = enabled
                    Toast.makeText(
                        context,
                        "Bounding Box: ${if (enabled) "Yes" else "No"}",
                        Toast.LENGTH_SHORT
                    ).show()
                    true
                }

            // ───── Take Photo Preference ─────
            findPreference<SwitchPreference>("take_photo")
                ?.setOnPreferenceChangeListener { _, newValue ->
                    val enabled = newValue as Boolean
                    Settings.ExportData.takePhoto = enabled
                    Toast.makeText(
                        context,
                        "Photo Saving: ${if (enabled) "Yes" else "No"}",
                        Toast.LENGTH_SHORT
                    ).show()
                    true
                }

            // ───── Export Video Data Preference ─────
            findPreference<SwitchPreference>("video_data")
                ?.setOnPreferenceChangeListener { _, newValue ->
                    val enabled = newValue as Boolean
                    Settings.ExportData.videoDATA = enabled
                    Toast.makeText(
                        context,
                        "Video Saving: ${if (enabled) "Yes" else "No"}",
                        Toast.LENGTH_SHORT
                    ).show()
                    true
                }
        }
    }

    override fun onBackPressed() {
        super.onBackPressed()
        setResult(RESULT_OK, Intent())
        finish()
    }
}