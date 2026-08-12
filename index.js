const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const ffmpeg = require("fluent-ffmpeg");

const app = express();
const PORT = 4000;

// JSON Data Receive කිරීමට
app.use(express.json());

// Uploads folder එක සෑදීම
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

// Direct Files Serve කිරීම
app.use("/files", express.static(uploadDir));

// ---------------------------------------------------------
// 1. DIRECT FILE UPLOAD (Image, Audio, Video - 1GB Limit)
// ---------------------------------------------------------
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, "uploads/"),
    filename: (req, file, cb) => {
        const uniqueName = `${Date.now()}_${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
        cb(null, uniqueName);
    }
});

const fileFilter = (req, file, cb) => {
    const fileType = file.mimetype.split("/")[0];
    if (["image", "audio", "video"].includes(fileType)) {
        cb(null, true);
    } else {
        cb(new Error("Only Image, Audio, and Video files are allowed!"), false);
    }
};

const upload = multer({ storage, fileFilter, limits: { fileSize: 1024 * 1024 * 1024 } });

app.post("/upload", upload.single("file"), (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ status: false, message: "No file uploaded" });

        const fileUrl = `${req.protocol}://${req.get("host")}/files/${req.file.filename}`;
        
        // තත්පර 50න් Delete කිරීම
        setTimeout(() => {
            if (fs.existsSync(req.file.path)) {
                fs.unlink(req.file.path, () => console.log(`Auto-deleted file: ${req.file.filename}`));
            }
        }, 50000);

        res.json({ status: true, type: req.file.mimetype.split("/")[0], url: fileUrl, expires_in: "50s" });
    } catch (error) {
        res.status(500).json({ status: false, message: error.message });
    }
});

// ---------------------------------------------------------
// 2. PROCESS LINK FROM MAIN SERVER (Download & Convert)
// ---------------------------------------------------------
app.post("/process-link", async (req, res) => {
    const { downloadUrl } = req.body;
    
    if (!downloadUrl) {
        return res.status(400).json({ status: false, message: "downloadUrl is required" });
    }

    const tempId = Date.now();
    const inputFilePath = path.join(__dirname, `input_${tempId}.mp4`);
    const finalFileName = `converted_${tempId}.mp4`;
    const finalFilePath = path.join(uploadDir, finalFileName);

    try {
        // Step A: URL එකෙන් වීඩියෝව Download කරගැනීම
        const writer = fs.createWriteStream(inputFilePath);
        const response = await axios({ url: downloadUrl, method: 'GET', responseType: 'stream' });
        response.data.pipe(writer);
        await new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });

        // Step B: FFmpeg මගින් Convert කිරීම
        await new Promise((resolve, reject) => {
            ffmpeg(inputFilePath)
                .output(finalFilePath)
                .videoCodec('libx264')
                .audioCodec('aac')
                .on('end', resolve)
                .on('error', reject)
                .run();
        });

        // Step C: Input File එක මකා දැමීම
        if (fs.existsSync(inputFilePath)) fs.unlinkSync(inputFilePath);

        // Step D: තත්පර 50න් Output File එක Delete වීමට සැකසීම
        setTimeout(() => {
            if (fs.existsSync(finalFilePath)) {
                fs.unlink(finalFilePath, () => console.log(`Auto-deleted converted video: ${finalFileName}`));
            }
        }, 50000);

        const fileUrl = `${req.protocol}://${req.get("host")}/files/${finalFileName}`;
        
        // Main Server එකට Link එක යැවීම
        res.json({ status: true, url: fileUrl, expires_in: "50s" });

    } catch (error) {
        if (fs.existsSync(inputFilePath)) fs.unlinkSync(inputFilePath);
        if (fs.existsSync(finalFilePath)) fs.unlinkSync(finalFilePath);
        res.status(500).json({ status: false, message: error.message });
    }
});

app.listen(PORT, () => console.log(`CDN/Upload Server running on port ${PORT}`));
