const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();

const PORT = process.env.PORT || 3000;

const MODEL_PATH = path.join(__dirname, "model", "model.onnx");
const UPLOAD_DIR = path.join(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

if (!fs.existsSync(path.dirname(MODEL_PATH))) {
    fs.mkdirSync(path.dirname(MODEL_PATH), { recursive: true });
}

const upload = multer({
    dest: UPLOAD_DIR,
    limits: {
        fileSize: 20 * 1024 * 1024
    }
});

let session;

async function loadModel() {
    console.log("Loading BiRefNet...");

    if (!fs.existsSync(MODEL_PATH)) {
        throw new Error(
            `Model not found:\n${MODEL_PATH}\n\nPut the BiRefNet ONNX model there.`
        );
    }

    session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["cpu"]
    });

    console.log("BiRefNet loaded.");
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);
}

function imageToTensor(raw, width, height) {
    const size = width * height;

    const r = new Float32Array(size);
    const g = new Float32Array(size);
    const b = new Float32Array(size);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < size; i++) {
        const p = i * 3;

        r[i] = (raw[p] / 255 - mean[0]) / std[0];
        g[i] = (raw[p + 1] / 255 - mean[1]) / std[1];
        b[i] = (raw[p + 2] / 255 - mean[2]) / std[2];
    }

    const data = new Float32Array(size * 3);

    data.set(r, 0);
    data.set(g, size);
    data.set(b, size * 2);

    return new ort.Tensor(
        "float32",
        data,
        [1, 3, height, width]
    );
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function findOutputTensor(results) {
    const names = Object.keys(results);

    if (!names.length) {
        throw new Error("Model returned no output.");
    }

    return results[names[names.length - 1]];
}

async function removeBackground(inputBuffer) {
    if (!session) {
        throw new Error("Model is not loaded.");
    }

    const original = await sharp(inputBuffer)
        .rotate()
        .ensureAlpha()
        .raw()
        .toBuffer({
            resolveWithObject: true
        });

    const originalWidth = original.info.width;
    const originalHeight = original.info.height;

    const SIZE = 1024;

    const resized = await sharp(inputBuffer)
        .rotate()
        .resize(SIZE, SIZE, {
            fit: "fill"
        })
        .removeAlpha()
        .raw()
        .toBuffer();

    const tensor = imageToTensor(
        resized,
        SIZE,
        SIZE
    );

    const inputName = session.inputNames[0];

    const results = await session.run({
        [inputName]: tensor
    });

    const output = findOutputTensor(results);

    const outputData = output.data;

    const mask = Buffer.alloc(
        SIZE * SIZE
    );

    for (let i = 0; i < SIZE * SIZE; i++) {
        let value = Number(outputData[i]);

        value = sigmoid(value);

        value = Math.max(
            0,
            Math.min(1, value)
        );

        mask[i] = Math.round(value * 255);
    }

    const maskBuffer = await sharp(mask, {
        raw: {
            width: SIZE,
            height: SIZE,
            channels: 1
        }
    })
        .resize(
            originalWidth,
            originalHeight
        )
        .png()
        .toBuffer();

    const rgb = await sharp(inputBuffer)
        .rotate()
        .ensureAlpha()
        .toBuffer();

    return await sharp(rgb)
        .joinChannel(maskBuffer, {
            raw: {
                width: originalWidth,
                height: originalHeight,
                channels: 1
            }
        })
        .png()
        .toBuffer();
}

app.get("/", (req, res) => {
    res.send(`
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BiRefNet Background Remover</title>

<style>
body {
    font-family: Arial;
    background: #111;
    color: white;
    max-width: 700px;
    margin: auto;
    padding: 30px;
}

.box {
    background: #222;
    padding: 25px;
    border-radius: 15px;
}

input, button {
    width: 100%;
    padding: 14px;
    margin-top: 12px;
    box-sizing: border-box;
}

button {
    cursor: pointer;
}

img {
    max-width: 100%;
    margin-top: 20px;
    border-radius: 10px;
}
</style>
</head>

<body>

<div class="box">

<h1>BiRefNet Background Remover</h1>

<form id="form">
<input
    type="file"
    id="image"
    accept="image/*"
    required
>

<button type="submit">
Remove Background
</button>
</form>

<p id="status"></p>

<img id="result" style="display:none">

<br>

<a id="download" style="display:none" download="removed-background.png">
Download PNG
</a>

</div>

<script>

document
.getElementById("form")
.addEventListener("submit", async (e) => {

    e.preventDefault();

    const file =
        document.getElementById("image").files[0];

    if (!file) return;

    const formData = new FormData();

    formData.append("image", file);

    document.getElementById("status").textContent =
        "Removing background...";

    try {

        const response = await fetch(
            "/remove-bg",
            {
                method: "POST",
                body: formData
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
            URL.createObjectURL(blob);

        const result =
            document.getElementById("result");

        result.src = url;
        result.style.display = "block";

        const download =
            document.getElementById("download");

        download.href = url;
        download.style.display = "block";

        document.getElementById("status")
            .textContent =
            "Background removed successfully.";

    } catch (err) {

        document.getElementById("status")
            .textContent =
            "Error: " + err.message;

    }

});

</script>

</body>
</html>
    `);
});

app.post(
    "/remove-bg",
    upload.single("image"),
    async (req, res) => {

        let inputPath;

        try {

            if (!req.file) {
                return res.status(400)
                    .send("Image is required.");
            }

            inputPath = req.file.path;

            const input =
                fs.readFileSync(inputPath);

            const output =
                await removeBackground(input);

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

            console.error(error);

            res.status(500).send(
                "Background removal failed: " +
                error.message
            );

        } finally {

            if (
                inputPath &&
                fs.existsSync(inputPath)
            ) {
                fs.unlinkSync(inputPath);
            }

        }
    }
);

app.post(
    "/remove-bg-url",
    express.json({
        limit: "2mb"
    }),
    async (req, res) => {

        try {

            const { url } = req.body;

            if (!url) {
                return res.status(400)
                    .json({
                        status: false,
                        error: "URL is required"
                    });
            }

            const response =
                await axios.get(url, {
                    responseType: "arraybuffer",
                    timeout: 30000,
                    maxContentLength:
                        20 * 1024 * 1024,
                    headers: {
                        "User-Agent":
                            "Mozilla/5.0"
                    }
                });

            const output =
                await removeBackground(
                    Buffer.from(response.data)
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

            console.error(error);

            res.status(500).json({
                status: false,
                error: error.message
            });

        }
    }
);

loadModel()
    .then(() => {

        app.listen(PORT, () => {

            console.log(
                `Server running on port ${PORT}`
            );

            console.log(
                `http://localhost:${PORT}`
            );

        });

    })
    .catch(error => {

        console.error(
            "Failed to start:",
            error
        );

        process.exit(1);

    });
