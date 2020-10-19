// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.laquysoft.custommodel

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.huawei.hms.mlsdk.common.MLException
import com.huawei.hms.mlsdk.custom.*
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.max
import kotlin.math.min


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {

    /** Data structure holding pairs of <label, confidence> for each inference result */
    data class LabelConfidence(val label: String, val confidence: Float)

    /** Current image being displayed in our app's screen */
    private var selectedImage: Bitmap? = null

    /** List of JPG files in our assets folder */
    private val imagePaths by lazy {
        resources.assets.list("")!!.filter { it.endsWith(".jpg") }
    }

    /** Labels corresponding to the output of the vision model. */
    private val labelList by lazy {
        BufferedReader(InputStreamReader(resources.assets.open(LABEL_PATH))).lineSequence().toList()
    }

    /** Preallocated buffers for storing image data. */
    private val imageBuffer = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

    // Gets the targeted width / height.
    private val targetedWidthHeight: Pair<Int, Int>
        get() {
            val targetWidth: Int
            val targetHeight: Int
            val maxWidthForPortraitMode = image_view.width
            val maxHeightForPortraitMode = image_view.height
            targetWidth = maxWidthForPortraitMode
            targetHeight = maxHeightForPortraitMode
            return Pair(targetWidth, targetHeight)
        }

    /** Input options used for our model interpreter */
    private val modelInputOutputSettings by lazy {
        //Input in format NCHW
        val inputDims = intArrayOf(1, 3, 224, 224)
        val outputDims = intArrayOf(1, 1001)
        MLModelInputOutputSettings.Factory()
            .setInputFormat(0, MLModelDataType.BYTE, inputDims)
            .setOutputFormat(0, MLModelDataType.BYTE, outputDims)
            .create()
    }


    /** Firebase model interpreter used for the local model from assets */
    private lateinit var modelExecutor: MLModelExecutor

    /** Initialize a local model interpreter from assets file */
    private fun createModelExecutor(): MLModelExecutor {
        val customModel = MLCustomLocalModel.Factory("cartoongan")
            .setAssetPathFile("cartoongan_fp16.ms")
            .create()

        val settings: MLModelExecutorSettings =
            MLModelExecutorSettings.Factory(customModel).create()
        return MLModelExecutor.getInstance(settings)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            imagePaths.mapIndexed { idx, _ -> "Image ${idx + 1}" })

        spinner.adapter = adapter
        spinner.onItemSelectedListener = this
        button_run.setOnClickListener { runInference() }

        // Disable the inference button until model is loaded
        button_run.isEnabled = false

        // Load the model interpreter in a coroutine
        lifecycleScope.launch(Dispatchers.IO) {
            //modelInterpreter = createLocalModelInterpreter()
            //modelInterpreter = createRemoteModelInterpreter()
            modelExecutor = createModelExecutor()
            runOnUiThread { button_run.isEnabled = true }
        }

    }

    /** Uses model to make predictions and interpret output into likely labels. */
    private fun runInference() = selectedImage?.let { image ->

        // Create input data.
        val imgData = convertBitmapToByteBuffer(image)


        try {
            // Create model inputs from our image data.
            val modelInputs = MLModelInputs.Factory().add(imgData).create()


            // Perform inference using our model interpreter.
            modelExecutor.exec(modelInputs, modelInputOutputSettings).continueWith {
                val inferenceOutput: Array<FloatArray> = it.result?.getOutput<Array<FloatArray>>(0)!!

                // Display labels on the screen using an overlay
                val topLabels = getTopLabels(inferenceOutput)
                graphic_overlay.clear()
                graphic_overlay.add(LabelGraphic(graphic_overlay, topLabels))
                topLabels
            }

        } catch (exc: MLException) {
            val msg = "Error running model inference"
            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
            Log.e(TAG, msg, exc)
        }
    }

    /** Gets the top labels in the results. */
    @Synchronized
    private fun getTopLabels(inferenceOutput: Array<FloatArray>): List<String> {
        // Since we ran inference on a single image, inference output will have a single row.
        val imageInference = inferenceOutput.first()

        return labelList.mapIndexed { idx, label ->
            LabelConfidence(label, (imageInference[idx]))

            // Sort the results in decreasing order of confidence and return only top 3.
        }.sortedBy { it.confidence }.reversed().map { "${it.label}:${it.confidence}" }
            .subList(0, min(labelList.size, RESULTS_TO_SHOW))
    }

    /** Writes Image data into a `ByteBuffer`. */
    @Synchronized
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): Array<Array<Array<ByteArray>>> {
        val inputBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val input = Array(1) {
            Array(3) {
                Array(224) {
                   ByteArray(224)
                }
            }
        }

        val batchNum = 0
        for (i in 0..223) {
            for (j in 0..223) {
                val pixel: Int = inputBitmap.getPixel(i, j)
                input[batchNum][0][j][i] = Color.red(pixel).toByte()
                input[batchNum][1][j][i] = Color.green(pixel).toByte()
                input[batchNum][2][j][i] = Color.blue(pixel).toByte()
            }
        }

        return input
    }

    override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
        graphic_overlay.clear()
        selectedImage = decodeBitmapAsset(this, imagePaths[position])
        if (selectedImage != null) {
            // Get the dimensions of the View
            val targetedSize = targetedWidthHeight

            val targetWidth = targetedSize.first
            val maxHeight = targetedSize.second

            // Determine how much to scale down the image
            val scaleFactor = max(
                selectedImage!!.width.toFloat() / targetWidth.toFloat(),
                selectedImage!!.height.toFloat() / maxHeight.toFloat()
            )

            val resizedBitmap = Bitmap.createScaledBitmap(
                selectedImage!!,
                (selectedImage!!.width / scaleFactor).toInt(),
                (selectedImage!!.height / scaleFactor).toInt(),
                true
            )

            image_view.setImageBitmap(resizedBitmap)
            selectedImage = resizedBitmap
        }
    }

    override fun onNothingSelected(parent: AdapterView<*>) = Unit

    companion object {
        private val TAG = MainActivity::class.java.simpleName

        /** Name of the label file stored in Assets. */
        private const val LABEL_PATH = "labels.txt"

        /** Name of the remote model in Firebase. */
        private const val REMOTE_MODEL_NAME = "mobilenet_v1_224_quant"

        /** Number of results to show in the UI. */
        private const val RESULTS_TO_SHOW = 3

        /** Dimensions of inputs. */
        private const val DIM_BATCH_SIZE = 1
        private const val DIM_PIXEL_SIZE = 3
        private const val DIM_IMG_SIZE_X = 224
        private const val DIM_IMG_SIZE_Y = 224

        /** Utility function for loading and resizing images from app asset folder. */
        fun decodeBitmapAsset(context: Context, filePath: String): Bitmap =
            context.assets.open(filePath).let { BitmapFactory.decodeStream(it) }
    }
}
