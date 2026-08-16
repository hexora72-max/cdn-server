const express = require("express");
const multer = require("multer");
const sharp = require("sharp");

const app = express();

const PORT = Number(process.env.PORT) || 3000;

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 20 * 1024 * 1024
    }
});

let model = null;
let processor = null;

let modelReady = false;
let modelError = null;
let loading = false;


/* =========================================================
   LOAD MODEL
========================================================= */

async function loadModel() {

    if (modelReady) {
        return;
    }

    if (loading) {
        return;
    }

    loading = true;
    modelError = null;

    try {

        console.log(
            "Loading Transformers.js..."
        );

        const {
            AutoModel,
            AutoProcessor
        } = await import(
            "@huggingface/transformers"
        );

        const MODEL =
            "onnx-community/BiRefNet-ONNX";

        console.log(
            "Loading processor..."
        );

        processor =
            await AutoProcessor.from_pretrained(
                MODEL
            );

        console.log(
            "Loading BiRefNet model..."
        );

        model =
            await AutoModel.from_pretrained(
                MODEL,
                {
                    dtype: "fp32"
                }
            );

        modelReady = true;

        console.log(
            "================================"
        );

        console.log(
            "BiRefNet READY"
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
            "MODEL ERROR:"
        );

        console.error(
            error
        );

    } finally {

        loading = false;
    }
}


/* =========================================================
   REMOVE BACKGROUND
========================================================= */

async function removeBackground(
    buffer
) {

    if (!modelReady) {

        throw new Error(
            modelError ||
            "BiRefNet model is still loading."
        );
    }

    const {
        RawImage
    } = await import(
        "@huggingface/transformers"
    );

    const image =
        await RawImage.read(
            buffer
        );

    const originalWidth =
        image.width;

    const originalHeight =
        image.height;

    console.log(
        "Input:",
        originalWidth,
        "x",
        originalHeight
    );

    const inputs =
        await processor(
            image
        );

    const output =
        await model(
            inputs
        );

    if (!output) {

        throw new Error(
            "BiRefNet returned no output."
        );
    }

    let maskTensor =
        output.output_image;

    if (!maskTensor) {

        throw new Error(
            "BiRefNet output_image not found."
        );
    }

    if (
        Array.isArray(maskTensor)
    ) {

        maskTensor =
            maskTensor[0];
    }

    let mask =
        RawImage.fromTensor(
            maskTensor
        );

    /*
     * Convert model output
     * to alpha mask.
     */

    const data =
        mask.data;

    for (
        let i = 0;
        i < data.length;
        i++
    ) {

        let value =
            Number(data[i]);

        /*
         * sigmoid
         */

        value =
            1 /
            (
                1 +
                Math.exp(-value)
            );

        value =
            Math.max(
                0,
                Math.min(
                    1,
                    value
                )
            );

        data[i] =
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
        await sharp(buffer)
            .rotate()
            .ensureAlpha()
            .toBuffer();

    /*
     * Apply mask as alpha.
     */

    const output =
        await sharp(original)
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
            modelLoaded: modelReady,
            loading: loading,
            error: modelError
        });

    }
);


/* =========================================================
   HOME
========================================================= */

app.get(
    "/",
    (req, res) => {

        res.type("html").send(`
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
    font-weight: bold;
    font-size: 16px;
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
    background: white;
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
    display: none;
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
Remove image background automatically.
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
Checking model...
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
        await fetch("/health");

        const data =
        await response.json();

        if (
            data.modelLoaded
        ) {

            status.textContent =
            "Ready.";

        } else if (
            data.loading
        ) {

            status.textContent =
            "AI model is loading. Please wait...";

            setTimeout(
                checkModel,
                3000
            );

        } else {

            status.textContent =
            "Model error: " +
            (
                data.error ||
                "Unknown error"
            );
        }

    } catch (error) {

        status.textContent =
        "Server error.";

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

    result.classList.add(
        "hidden"
    );

    download.classList.add(
        "hidden"
    );

    status.textContent =
    "Removing background...";


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
                method: "POST",
                body: formData
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

        console.error(
            error
        );

        status.textContent =
        error.message;

    } finally {

        button.disabled =
        false;
    }

});


checkModel();

</script>

</body>

</html>
        `);

    }
);


/* =========================================================
   REMOVE BG API
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
                        "BiRefNet is still loading. Try again."
                });
            }

            console.log(
                "Processing image..."
            );

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
                "REMOVE BG ERROR:"
            );

            console.error(
                error
            );

            if (
                !res.headersSent
            ) {

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
   START SERVER FIRST
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
            `PORT: ${PORT}`
        );

        console.log(
            "================================"
        );

        /*
         * IMPORTANT:
         * Do NOT await model here.
         * Railway gets the port immediately.
         */

        loadModel();

    }
);
