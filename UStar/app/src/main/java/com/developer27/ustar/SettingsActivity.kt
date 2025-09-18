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

/**
 * SettingsActivity
 * ----------------
 * Hosts a [PreferenceFragmentCompat] to expose runtime toggles backed by the app's
 * in-memory [Settings] singleton (videoprocessing module).
 *
 * Behavior summary:
 *  - Locks the screen to portrait and keeps it on while this screen is visible.
 *  - Loads `R.layout.settings_activity`, then injects [SettingsFragment] into `R.id.settings_container`.
 *  - When the user presses back, returns RESULT_OK to the caller (handy if the caller wants to refresh UI).
 *
 * Note on persistence:
 *  - The preferences defined in `R.xml.root_preferences` are typically persisted by the
 *    AndroidX Preferences library to SharedPreferences.
 *  - This fragment ALSO mirrors each toggle into the in-memory [Settings] object so changes
 *    take effect immediately at runtime. If you want Settings to reflect stored values on app launch,
 *    initialize [Settings] from SharedPreferences early in your app (e.g., in Application.onCreate()).
 */
class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Lock to portrait & keep screen awake while adjusting settings.
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Inflate the container layout that holds the preference fragment.
        setContentView(R.layout.settings_activity)

        // Insert the SettingsFragment into the placeholder container.
        supportFragmentManager
            .beginTransaction()
            .replace(R.id.settings_container, SettingsFragment())
            .commit()
    }

    /**
     * Fragment that inflates preferences from XML and wires each toggle to the in-memory Settings.
     *
     * Keys expected in R.xml.root_preferences:
     *  - "enable_bounding_box" -> toggles drawing overlays.
     *  - "take_photo"          -> toggles Take Photo button visibility/behavior.
     *  - "video_data"          -> toggles recording/export of processed video frames.
     */
    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            // Inflate the preference hierarchy from XML.
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // ───── Bounding Box Preference ─────
            // When toggled, immediately reflect the new value into Settings and show a toast.
            findPreference<SwitchPreference>("enable_bounding_box")
                ?.setOnPreferenceChangeListener { _, newValue ->
                    val enabled = newValue as Boolean
                    Settings.BoundingBox.enableBoundingBox = enabled
                    Toast.makeText(
                        context,
                        "Bounding Box: ${if (enabled) "Yes" else "No"}",
                        Toast.LENGTH_SHORT
                    ).show()
                    true // returning true lets the Preference library persist the new value
                }

            // ───── Take Photo Preference ─────
            // Mirrors toggle into Settings.ExportData.takePhoto (UI reacts on next resume or immediately where observed).
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
            // When enabled, MainActivity will start a ProcessedVideoRecorder on Start Tracking.
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

    /**
     * Return to the caller with RESULT_OK so the previous screen can react (e.g., refresh UI).
     * NOTE: For newer APIs, consider using OnBackPressedDispatcher; this remains fine for simple flows.
     */
    override fun onBackPressed() {
        super.onBackPressed()
        setResult(RESULT_OK, Intent())
        finish()
    }
}