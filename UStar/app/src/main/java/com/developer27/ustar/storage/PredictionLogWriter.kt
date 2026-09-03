package com.developer27.ustar.storage

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object PredictionLogWriter {
    private const val FILE_NAME = "UStar_Cube_Prediction.txt"
    private const val TAG = "UStarLogger"

    fun write(context: Context, logBody: CharSequence): Boolean {
        val timestamp = SimpleDateFormat(
            "yyyy-MM-dd HH:mm:ss",
            Locale.getDefault()
        ).format(Date())
        val contents = buildString {
            appendLine("UStar UIOD Tag Features")
            appendLine("Prediction Date: $timestamp")
            append(logBody)
        }

        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                writeToMediaStore(context, contents)
            } else {
                writeToAppDocuments(context, contents)
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Unable to write prediction log", e)
            false
        }
    }

    private fun writeToMediaStore(context: Context, contents: String) {
        val resolver = context.contentResolver
        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val relativePath = "${Environment.DIRECTORY_DOCUMENTS}/"
        val existing = resolver.query(
            collection,
            arrayOf(MediaStore.MediaColumns._ID),
            "${MediaStore.MediaColumns.DISPLAY_NAME} = ? AND " +
                "${MediaStore.MediaColumns.RELATIVE_PATH} = ?",
            arrayOf(FILE_NAME, relativePath),
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                ContentUris.withAppendedId(collection, cursor.getLong(0))
            } else {
                null
            }
        }

        var inserted: Uri? = null
        val uri = existing ?: resolver.insert(
            collection,
            ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, FILE_NAME)
                put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
                put(MediaStore.MediaColumns.RELATIVE_PATH, relativePath)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        )?.also { inserted = it }
            ?: error("MediaStore did not create the prediction log")

        try {
            resolver.openOutputStream(uri, "wt")?.bufferedWriter()?.use { writer ->
                writer.write(contents)
            } ?: error("MediaStore did not open the prediction log")

            if (inserted != null) {
                resolver.update(
                    uri,
                    ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) },
                    null,
                    null
                )
            }
        } catch (e: Exception) {
            inserted?.let { resolver.delete(it, null, null) }
            throw e
        }

        Log.i(TAG, "Prediction log saved to Documents/$FILE_NAME")
    }

    private fun writeToAppDocuments(context: Context, contents: String) {
        val directory = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            ?: context.filesDir
        val file = File(directory, FILE_NAME)
        file.parentFile?.mkdirs()
        file.writeText(contents)
        Log.i(TAG, "Prediction log saved to ${file.absolutePath}")
    }
}
