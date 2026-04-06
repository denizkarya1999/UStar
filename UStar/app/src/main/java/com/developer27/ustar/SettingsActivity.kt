package com.developer27.ustar

import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.developer27.ustar.extras.LocalPictureInferenceActivity
import com.developer27.ustar.extras.LocalPhotoInferenceDynaSpaActivity
import com.developer27.ustar.videoprocessing.Settings

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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

            // New radio-style selection preference
            val denoiserPref: ListPreference? =
                findPreference("pref_denoiser_type")

            // Existing CycleGAN option
            val localPicturePref: Preference? =
                findPreference("pref_local_picture_inference")

            // Existing DynaSpa option
            val localDynaSpaPref: Preference? =
                findPreference("pref_local_dynaspa_inference")

            // Initialize runtime value from preference
            Settings.selectedDenoiser = denoiserPref?.value ?: "cyclegan"

            // Update runtime value when changed
            denoiserPref?.setOnPreferenceChangeListener { _, newValue ->
                Settings.selectedDenoiser = newValue.toString()
                true
            }

            // Existing CycleGAN activity
            localPicturePref?.setOnPreferenceClickListener {
                val ctx = requireContext()
                val intent = Intent(ctx, LocalPictureInferenceActivity::class.java)
                startActivity(intent)
                true
            }

            // Existing DynaSpa activity
            localDynaSpaPref?.setOnPreferenceClickListener {
                val ctx = requireContext()
                val intent = Intent(ctx, LocalPhotoInferenceDynaSpaActivity::class.java)
                startActivity(intent)
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