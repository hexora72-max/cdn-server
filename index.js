const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();

const PORT = process.env.PORT || 3000;

const MODEL_DIR = path.join(__dirname, "model");
const MODEL_PATH = path.join(MODEL_DIR, "model.onnx");
const UPLOAD_DIR = path.join(__dirname, "uploads");

const MODEL_URL =
    "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/onnx/model.onnx";

let session = null;

fs.mkdirSync(MODEL_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({
    dest: UPLOAD_DIR,
    limits: {
        fileSize: 20 * 1024 * 1024
    }
});


/* =========================================================
   DOWNLOAD MODEL
========================================================= */

async function downloadModel() {

    if (fs.existsSync(MODEL_PATH)) {

        const stats =
            fs.statSync(MODEL_PATH);

        if (
            stats.size >
            100 * 1024 * 1024
        ) {

            console.log(
                "BiRefNet model already exists:",
                (
                    stats.size /
                    1024 /
                    1024
                ).toFixed(2),
                "MB"
            );

            return;
        }

        console.log(
            "Incomplete model found. Removing..."
        );

        fs.unlinkSync(
            MODEL_PATH
        );
    }

    const tempPath =
        MODEL_PATH + ".tmp";

    if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
    }

    console.log("");
    console.log(
        "======================================"
    );
    console.log(
        "Downloading BiRefNet ONNX model..."
    );
    console.log(
        "======================================"
    );
    console.log("");

    const response =
        await axios({
            method: "GET",
            url: MODEL_URL,
            responseType: "stream",
            timeout: 0,
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                "User-Agent":
                    "Mozilla/5.0"
            }
        });

    const total =
        Number(
            response.headers[
                "content-length"
            ]
        ) || 0;

    let downloaded = 0;
    let lastPercent = -1;

    response.data.on(
        "data",
        chunk => {

            downloaded +=
                chunk.length;

            if (total > 0) {

                const percent =
                    Math.floor(
                        (
                            downloaded /
                            total
                        ) * 100
                    );

                if (
                    percent !==
                    lastPercent
                ) {

                    lastPercent =
                        percent;

                    process.stdout.write(
                        `\rModel download: ${percent}%`
                    );
                }
            }
        }
    );

    const writer =
        fs.createWriteStream(
            tempPath
        );

    response.data.pipe(
        writer
    );

    await new Promise(
        (resolve, reject) => {

            writer.on(
                "finish",
                resolve
            );

            writer.on(
                "error",
                reject
            );

            response.data.on(
                "error",
                reject
            );
        }
    );

    console.log("");

    const stats =
        fs.statSync(
            tempPath
        );

    if (
        stats.size <
        100 * 1024 * 1024
    ) {

        fs.unlinkSync(
            tempPath
        );

        throw new Error(
            "BiRefNet model download is incomplete."
        );
    }

    fs.renameSync(
        tempPath,
        MODEL_PATH
    );

    console.log(
        "Model downloaded successfully."
    );

    console.log(
        "Model size:",
        (
            stats.size /
            1024 /
            1024
        ).toFixed(2),
        "MB"
    );
}


/* =========================================================
   LOAD MODEL
========================================================= */

async function loadModel() {

    await downloadModel();

    console.log(
        "Loading BiRefNet..."
    );

    session =
        await ort.InferenceSession.create(
            MODEL_PATH,
            {
                executionProviders: [
                    "cpu"
                ]
            }
        );

    console.log(
        "BiRefNet loaded successfully."
    );

    console.log(
        "Inputs:",
        session.inputNames
    );

    console.log(
        "Outputs:",
        session.outputNames
    );
}


/* =========================================================
   IMAGE TO TENSOR
========================================================= */

function imageToTensor(
    raw,
    width,
    height
) {

    const size =
        width * height;

    const data =
        new Float32Array(
            size * 3
        );

    const mean = [
        0.485,
        0.456,
        0.406
    ];

    const std = [
        0.229,
        0.224,
        0.225
    ];

    for (
        let i = 0;
        i < size;
        i++
    ) {

        const p =
            i * 3;

        const r =
            raw[p] / 255;

        const g =
            raw[p + 1] / 255;

        const b =
            raw[p + 2] / 255;

        data[i] =
            (r - mean[0]) /
            std[0];

        data[
            size + i
        ] =
            (g - mean[1]) /
            std[1];

        data[
            size * 2 + i
        ] =
            (b - mean[2]) /
            std[2];
    }

    return new ort.Tensor(
        "float32",
        data,
        [
            1,
            3,
            height,
            width
        ]
    );
}


/* =========================================================
   SIGMOID
========================================================= */

function sigmoid(value) {

    return 1 /
        (
            1 +
            Math.exp(-value)
        );
}


/* =========================================================
   GET MASK
========================================================= */

function getMaskOutput(
    results
) {

    const names =
        session.outputNames;

    if (!names.length) {
        throw new Error(
            "BiRefNet returned no outputs."
        );
    }

    for (
        const name of names
    ) {

        const tensor =
            results[name];

        if (
            !tensor ||
            !tensor.data
        ) {
            continue;
        }

        const dims =
            tensor.dims || [];

        if (
            dims.length === 4 &&
            dims[0] === 1 &&
            dims[1] === 1
        ) {

            return tensor;
        }
    }

    const last =
        results[
            names[
                names.length - 1
            ]
        ];

    if (
        !last ||
        !last.data
    ) {

        throw new Error(
            "Could not find mask output."
        );
    }

    return last;
}


/* =========================================================
   REMOVE BACKGROUND
========================================================= */

async function removeBackground(
    inputBuffer
) {

    if (!session) {

        throw new Error(
            "BiRefNet model is not loaded."
        );
    }

    const metadata =
        await sharp(inputBuffer)
            .rotate()
            .metadata();

    const width =
        metadata.width;

    const height =
        metadata.height;

    if (
        !width ||
        !height
    ) {

        throw new Error(
            "Invalid image."
        );
    }

    const SIZE = 1024;

    const resized =
        await sharp(inputBuffer)
            .rotate()
            .resize(
                SIZE,
                SIZE,
                {
                    fit: "fill"
                }
            )
            .removeAlpha()
            .raw()
            .toBuffer();

    const tensor =
        imageToTensor(
            resized,
            SIZE,
            SIZE
        );

    const inputName =
        session.inputNames[0];

    const results =
        await session.run({
            [inputName]:
                tensor
        });

    const output =
        getMaskOutput(
            results
        );

    console.log(
        "Mask dimensions:",
        output.dims
    );

    const dims =
        output.dims || [];

    let maskWidth =
        SIZE;

    let maskHeight =
        SIZE;

    if (
        dims.length >= 2
    ) {

        maskHeight =
            dims[
                dims.length - 2
            ];

        maskWidth =
            dims[
                dims.length - 1
            ];
    }

    const pixelCount =
        maskWidth *
        maskHeight;

    const outputData =
        output.data;

    if (
        outputData.length <
        pixelCount
    ) {

        throw new Error(
            "Invalid BiRefNet mask output."
        );
    }

    const mask =
        Buffer.alloc(
            pixelCount
        );

    for (
        let i = 0;
        i < pixelCount;
        i++
    ) {

        let value =
            Number(
                outputData[i]
            );

        value =
            sigmoid(value);

        value =
            Math.max(
                0,
                Math.min(
                    1,
                    value
                )
            );

        mask[i] =
            Math.round(
                value * 255
            );
    }

    const maskBuffer =
        await sharp(
            mask,
            {
                raw: {
                    width:
                        maskWidth,
                    height:
                        maskHeight,
                    channels: 1
                }
            }
        )
            .resize(
                width,
                height,
                {
                    fit: "fill",
                    kernel:
                        sharp
                            .kernel
                            .lanczos3
                }
            )
            .png()
            .toBuffer();

    const result =
        await sharp(inputBuffer)
            .rotate()
            .ensureAlpha()
            .joinChannel(
                maskBuffer,
                {
                    raw: {
                        width,
                        height,
                        channels: 1
                    }
                }
            )
            .png()
            .toBuffer();

    return result;
}


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
    padding: 30px;
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
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
    margin-top: 20px;
    background: #333;
    color: white;
    border-radius: 10px;
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
    opacity: 0.5;
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
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    border-radius: 10px;
    background: white;
    color: black;
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
Select an image.
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
    document.getElementById(
        "form"
    );

const input =
    document.getElementById(
        "image"
    );

const button =
    document.getElementById(
        "button"
    );

const status =
    document.getElementById(
        "status"
    );

const result =
    document.getElementById(
        "result"
    );

const download =
    document.getElementById(
        "download"
    );

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

        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );

        try {

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

                throw new Error(
                    await response.text()
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

            status.textContent =
                "Error: " +
                error.message;

        } finally {

            button.disabled =
                false;
        }
    }
);

</script>

</body>

</html>
        `);
    }
);


/* =========================================================
   UPLOAD API
========================================================= */

app.post(
    "/remove-bg",
    upload.single("image"),

    async (req, res) => {

        let inputPath = null;

        try {

            if (!req.file) {

                return res
                    .status(400)
                    .send(
                        "Image is required."
                    );
            }

            inputPath =
                req.file.path;

            const input =
                await fs.promises.readFile(
                    inputPath
                );

            const output =
                await removeBackground(
                    input
                );

            res.setHeader(
                "Content-Type",
                "image/png"
            );

            res.setHeader(
                "Content-Disposition",
                'attachment; filename="removed-background.png"'
            );

            res.send(output);

        } catch (error) {

            console.error(
                "Background removal error:",
                error
            );

            res
                .status(500)
                .send(
                    "Background removal failed: " +
                    error.message
                );

        } finally {

            if (
                inputPath &&
                fs.existsSync(
                    inputPath
                )
            ) {

                fs.unlinkSync(
                    inputPath
                );
            }
        }
    }
);


/* =========================================================
   URL API
========================================================= */

app.use(
    express.json({
        limit: "2mb"
    })
);

app.post(
    "/remove-bg-url",

    async (req, res) => {

        try {

            const url =
                req.body.url;

            if (!url) {

                return res
                    .status(400)
                    .json({
                        status: false,
                        error:
                            "Image URL is required."
                    });
            }

            /*
             * Correct regex:
             * /^https?:\/\//i
             */

            if (
                !/^https?:\/\//i.test(
                    url
                )
            ) {

                return res
                    .status(400)
                    .json({
                        status: false,
                        error:
                            "Invalid image URL."
                    });
            }

            const response =
                await axios.get(
                    url,
                    {
                        responseType:
                            "arraybuffer",

                        timeout:
                            60000,

                        maxContentLength:
                            20 * 1024 * 1024,

                        maxBodyLength:
                            20 * 1024 * 1024,

                        headers: {
                            "User-Agent":
                                "Mozilla/5.0"
                        }
                    }
                );

            const output =
                await removeBackground(
                    Buffer.from(
                        response.data
                    )
                );

            res.setHeader(
                "Content-Type",
                "image/png"
            );

            res.setHeader(
                "Content-Disposition",
                'attachment; filename="removed-background.png"'
            );

            res.send(output);

        } catch (error) {

            console.error(
                "URL background removal error:",
                error
            );

            res
                .status(500)
                .json({
                    status: false,
                    error:
                        error.message
                });
        }
    }
);


/* =========================================================
   HEALTH
========================================================= */

app.get(
    "/health",
    (req, res) => {

        res.json({
            status: true,
            modelLoaded:
                !!session
        });

    }
);


/* =========================================================
   START
========================================================= */

async function start() {

    try {

        console.log(
            "Starting BiRefNet..."
        );

        await loadModel();

        app.listen(
            PORT,
            "0.0.0.0",
            () => {

                console.log("");
                console.log(
                    "================================"
                );

                console.log(
                    `Server running on port ${PORT}`
                );

                console.log(
                    "Background remover is ready."
                );

                console.log(
                    "================================"
                );
            }
        );

    } catch (error) {

        console.error(
            "Failed to start:"
        );

        console.error(
            error
        );

        process.exit(1);
    }
}

start();
