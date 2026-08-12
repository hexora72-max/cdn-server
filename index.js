const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 4000;

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "uploads/");
    },
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

const upload = multer({ 
    storage, 
    fileFilter,
    limits: { fileSize: 1024 * 1024 * 1024 }
});

app.use("/files", express.static(uploadDir));

app.post("/upload", upload.single("file"), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ status: false, message: "No file uploaded" });
        }

        const fileUrl = `${req.protocol}://${req.get("host")}/files/${req.file.filename}`;
        const filePath = req.file.path;
        
        setTimeout(() => {
            if (fs.existsSync(filePath)) {
                fs.unlink(filePath, (err) => {
                    if (err) {
                        console.error(`Error deleting file ${req.file.filename}:`, err);
                    } else {
                        console.log(`Auto-deleted file: ${req.file.filename} (after 50s)`);
                    }
                });
            }
        }, 50000);

        res.json({
            status: true,
            type: req.file.mimetype.split("/")[0],
            url: fileUrl,
            expires_in: "50 seconds"
        });
    } catch (error) {
        res.status(500).json({ status: false, message: error.message });
    }
});

app.use((err, req, res, next) => {
    if (err) {
        return res.status(400).json({ status: false, message: err.message });
    }
    next();
});

app.listen(PORT, () => console.log(`CDN Storage Server running on port ${PORT}`));
