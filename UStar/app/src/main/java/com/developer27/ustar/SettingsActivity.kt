package com.developer27.ustar

import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.developer27.ustar.extras.LocalPictureInferenceActivity
import com.developer27.ustar.extras.LocalPhotoInferenceDynaSpaActivity
import com.developer27.ustar.videoprocessing.Settings

/**
 * SettingsActivity
 * ----------------
 * Hosts a [PreferenceFragmentCompat] to expose runtime toggles backed by the app's
 * in-memory [Settings] singleton (videoprocessing module).
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
     */
    class SettingsFragment : PreferenceFragmentCompat() {
        override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
            // Inflate the preference hierarchy from XML.
            setPreferencesFromResource(R.xml.root_preferences, rootKey)

            // Action preference: Local Picture Inference (CycleGAN)
            val localPicturePref: Preference? =
                findPreference("pref_local_picture_inference")

            // Action preference: Local Picture Inference (DynaSpa)
            val localDynaSpaPref: Preference? =
                findPreference("pref_local_dynaspa_inference")

            // Launch CycleGAN activity
            localPicturePref?.setOnPreferenceClickListener {
                val ctx = requireContext()
                val intent = Intent(ctx, LocalPictureInferenceActivity::class.java)
                startActivity(intent)
                true
            }

            // Launch DynaSpa activity
            localDynaSpaPref?.setOnPreferenceClickListener {
                val ctx = requireContext()
                val intent = Intent(ctx, LocalPhotoInferenceDynaSpaActivity::class.java)
                startActivity(intent)
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