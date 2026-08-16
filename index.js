const express = require("express");
const multer = require("multer");
const sharp = require("sharp");

const app = express();

const PORT = Number(process.env.PORT) || 3000;

let model = null;
let processor = null;

let modelReady = false;
let modelLoading = false;
let modelError = null;

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 20 * 1024 * 1024
    }
});


/* =========================================================
   LOAD BIREfNET
========================================================= */

async function loadModel() {

    if (modelReady || modelLoading) {
        return;
    }

    modelLoading = true;
    modelError = null;

    try {

        console.log("Loading Transformers.js...");

        const {
            AutoModel,
            AutoProcessor
        } = await import("@huggingface/transformers");

        const modelName =
            "onnx-community/BiRefNet-ONNX";

        console.log(
            "Loading processor..."
        );

        processor =
            await AutoProcessor.from_pretrained(
                modelName
            );

        console.log(
            "Loading BiRefNet model..."
        );

        model =
            await AutoModel.from_pretrained(
                modelName,
                {
                    dtype: "fp32"
                }
            );

        modelReady = true;

        console.log(
            "================================"
        );

        console.log(
            "BiRefNet loaded successfully"
        );

        console.log(
            "================================"
        );

    } catch (error) {

        modelReady = false;

        modelError =
            error?.message ||
            String(error);

        console.error(
            "BiRefNet loading failed:"
        );

        console.error(
            error
        );

    } finally {

        modelLoading = false;
    }
}


/* =========================================================
   REMOVE BACKGROUND
========================================================= */

async function removeBackground(
    inputBuffer
) {

    if (!modelReady) {

        throw new Error(
            modelError ||
            "BiRefNet is still loading."
        );
    }

    const {
        RawImage
    } = await import(
        "@huggingface/transformers"
    );

    const image =
        await RawImage.read(
            inputBuffer
        );

    const originalWidth =
        image.width;

    const originalHeight =
        image.height;

    console.log(
        "Image:",
        originalWidth,
        "x",
        originalHeight
    );

    const inputs =
        await processor(
            image
        );

    const result =
        await model(
            inputs
        );

    console.log(
        "Model output:",
        Object.keys(result)
    );

    /*
     * BiRefNet output can differ
     * between model versions.
     */

    let maskTensor = null;

    if (result.output_image) {

        maskTensor =
            result.output_image;

    } else if (result.logits) {

        maskTensor =
            result.logits;

    } else {

        const keys =
            Object.keys(result);

        if (keys.length > 0) {

            maskTensor =
                result[keys[0]];
        }
    }

    if (!maskTensor) {

        throw new Error(
            "Could not find BiRefNet mask output."
        );
    }

    if (
        Array.isArray(maskTensor)
    ) {

        maskTensor =
            maskTensor[0];
    }

    let mask;

    try {

        mask =
            RawImage.fromTensor(
                maskTensor
            );

    } catch (error) {

        console.error(
            "Tensor conversion error:",
            error
        );

        throw new Error(
            "Could not convert BiRefNet output to mask."
        );
    }

    /*
     * Convert model mask values
     * into 0-255 alpha values.
     */

    const maskData =
        mask.data;

    for (
        let i = 0;
        i < maskData.length;
        i++
    ) {

        let value =
            Number(
                maskData[i]
            );

        /*
         * If output is already 0-1,
         * don't apply sigmoid.
         */

        if (
            value < 0 ||
            value > 1
        ) {

            value =
                1 /
                (
                    1 +
                    Math.exp(-value)
                );
        }

        value =
            Math.max(
                0,
                Math.min(
                    1,
                    value
                )
            );

        maskData[i] =
            Math.round(
                value * 255
            );
    }

    /*
     * Convert mask to PNG.
     */

    const maskPng =
        await mask
            .convert(1)
            .resize(
                originalWidth,
                originalHeight
            )
            .toSharp()
            .png()
            .toBuffer();

    /*
     * Original image.
     */

    const original =
        await sharp(
            inputBuffer
        )
            .rotate()
            .ensureAlpha()
            .toBuffer();

    /*
     * Apply mask as alpha channel.
     */

    const output =
        await sharp(
            original
        )
            .joinChannel(
                maskPng,
                {
                    raw: {
                        width:
                            originalWidth,
                        height:
                            originalHeight,
                        channels: 1
                    }
                }
            )
            .png()
            .toBuffer();

    return output;
}


/* =========================================================
   HEALTH
========================================================= */

app.get(
    "/health",
    (req, res) => {

        res.json({
            status: true,
            server: true,
            modelLoaded:
                modelReady,
            modelLoading:
                modelLoading,
            modelError:
                modelError
        });

    }
);


/* =========================================================
   HOME
========================================================= */

app.get(
    "/",
    (req, res) => {

        res.send(`
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1"
>

<title>
BiRefNet Background Remover
</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 25px;
    background: #111;
    color: white;
    font-family: Arial;
}

.container {
    max-width: 700px;
    margin: auto;
}

.card {
    background: #222;
    padding: 25px;
    border-radius: 18px;
}

h1 {
    text-align: center;
}

input {
    width: 100%;
    padding: 15px;
    margin-top: 15px;
    border-radius: 10px;
    background: #333;
    color: white;
}

button {
    width: 100%;
    padding: 15px;
    margin-top: 15px;
    border: 0;
    border-radius: 10px;
    background: white;
    color: black;
    font-size: 16px;
    font-weight: bold;
}

button:disabled {
    opacity: .5;
}

#status {
    text-align: center;
    margin-top: 20px;
}

#result {
    width: 100%;
    margin-top: 20px;
    border-radius: 10px;

    background-color: white;

    background-image:
        linear-gradient(
            45deg,
            #ccc 25%,
            transparent 25%
        ),
        linear-gradient(
            -45deg,
            #ccc 25%,
            transparent 25%
        ),
        linear-gradient(
            45deg,
            transparent 75%,
            #ccc 75%
        ),
        linear-gradient(
            -45deg,
            transparent 75%,
            #ccc 75%
        );

    background-size: 20px 20px;

    background-position:
        0 0,
        0 10px,
        10px -10px,
        -10px 0;
}

#download {
    display: block;
    margin-top: 20px;
    padding: 15px;
    text-align: center;
    background: white;
    color: black;
    border-radius: 10px;
    text-decoration: none;
    font-weight: bold;
}

.hidden {
    display: none !important;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>
BiRefNet
</h1>

<p>
AI Background Remover
</p>

<form id="form">

<input
id="image"
type="file"
accept="image/*"
required
>

<button
id="button"
type="submit"
>
Remove Background
</button>

</form>

<p id="status">
Loading AI model...
</p>

<img
id="result"
class="hidden"
>

<a
id="download"
class="hidden"
download="removed-background.png"
>
Download PNG
</a>

</div>

</div>

<script>

const form =
document.getElementById("form");

const input =
document.getElementById("image");

const button =
document.getElementById("button");

const status =
document.getElementById("status");

const result =
document.getElementById("result");

const download =
document.getElementById("download");


async function checkModel() {

    try {

        const response =
            await fetch(
                "/health"
            );

        const data =
            await response.json();

        if (
            data.modelLoaded
        ) {

            status.textContent =
                "Ready.";

            return;
        }

        if (
            data.modelLoading
        ) {

            status.textContent =
                "AI model loading...";

            setTimeout(
                checkModel,
                3000
            );

            return;
        }

        if (
            data.modelError
        ) {

            status.textContent =
                "Model error: " +
                data.modelError;

            setTimeout(
                checkModel,
                5000
            );

            return;
        }

        status.textContent =
            "Starting AI model...";

        setTimeout(
            checkModel,
            3000
        );

    } catch (error) {

        status.textContent =
            "Checking server...";

        setTimeout(
            checkModel,
            3000
        );
    }
}


form.addEventListener(
    "submit",
    async event => {

        event.preventDefault();

        const file =
            input.files[0];

        if (!file) {
            return;
        }

        button.disabled =
            true;

        status.textContent =
            "Removing background...";

        result.classList.add(
            "hidden"
        );

        download.classList.add(
            "hidden"
        );

        try {

            const formData =
                new FormData();

            formData.append(
                "image",
                file
            );

            const response =
                await fetch(
                    "/remove-bg",
                    {
                        method:
                            "POST",
                        body:
                            formData
                    }
                );

            if (!response.ok) {

                const text =
                    await response.text();

                throw new Error(
                    text
                );
            }

            const blob =
                await response.blob();

            const url =
                URL.createObjectURL(
                    blob
                );

            result.src =
                url;

            result.classList.remove(
                "hidden"
            );

            download.href =
                url;

            download.classList.remove(
                "hidden"
            );

            status.textContent =
                "Background removed successfully.";

        } catch (error) {

            console.error(error);

            status.textContent =
                "Error: " +
                error.message;

        } finally {

            button.disabled =
                false;
        }
    }
);


checkModel();

</script>

</body>

</html>
        `);

    }
);


/* =========================================================
   REMOVE BACKGROUND API
========================================================= */

app.post(
    "/remove-bg",
    upload.single("image"),

    async (req, res) => {

        try {

            if (!req.file) {

                return res
                    .status(400)
                    .json({
                        status: false,
                        error:
                            "Image is required."
                    });
            }

            if (!modelReady) {

                return res
                    .status(503)
                    .json({
                        status: false,
                        error:
                            modelError ||
                            "AI model is still loading. Try again."
                    });
            }

            const output =
                await removeBackground(
                    req.file.buffer
                );

            res.setHeader(
                "Content-Type",
                "image/png"
            );

            res.setHeader(
                "Content-Disposition",
                'inline; filename="removed-background.png"'
            );

            res.end(
                output
            );

        } catch (error) {

            console.error(
                "Background removal error:"
            );

            console.error(
                error
            );

            if (!res.headersSent) {

                res
                    .status(500)
                    .json({
                        status: false,
                        error:
                            error.message
                    });
            }
        }
    }
);


/* =========================================================
   START SERVER
========================================================= */

app.listen(
    PORT,
    "0.0.0.0",
    () => {

        console.log(
            "================================"
        );

        console.log(
            "BiRefNet server started"
        );

        console.log(
            "PORT:",
            PORT
        );

        console.log(
            "================================"
        );

        /*
         * Start model loading after
         * Railway has a listening port.
         */

        loadModel();

    }
);
