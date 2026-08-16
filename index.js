const express = require('express');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const app = express();
app.use(express.json());

const rmbgDir = path.join(__dirname, 'rmbg');
if (!fs.existsSync(rmbgDir)) {
    fs.mkdirSync(rmbgDir);
}

app.use('/rmbg', express.static(rmbgDir));

app.all('/api/remove-bg', async (req, res) => {
    let imageUrl = '';

    if (req.query && req.query.url) {
        const urlStartIndex = req.originalUrl.indexOf('url=') + 4;
        imageUrl = req.originalUrl.substring(urlStartIndex);
        imageUrl = decodeURIComponent(imageUrl);
    } 
    else if (req.body && req.body.url) {
        imageUrl = req.body.url;
    }

    if (!imageUrl) {
        return res.status(400).json({ 
            status: false, 
            owner: "@KingPoddaModz", 
            error: "Please provide an image url" 
        });
    }

    try {
        const hostUrl = `${req.protocol}://${req.get('host')}`;
        const finalUrl = await processAndSaveImage(imageUrl, hostUrl);
        
        if (finalUrl) {
            res.json({ 
                status: true, 
                owner: "@KingPoddaModz",
                url: finalUrl 
            });
        } else {
            res.status(500).json({ 
                status: false, 
                owner: "@KingPoddaModz",
                error: "Failed to process image" 
            });
        }
    } catch (error) {
        res.status(500).json({ 
            status: false, 
            owner: "@KingPoddaModz",
            error: error.message 
        });
    }
});

async function processAndSaveImage(imageUrl, hostUrl) {
    const sessionHash = crypto.randomBytes(5).toString('hex');
    const baseUrl = 'https://briaai-bria-rmbg-1-4.hf.space';

    try {
        const sourceImageRes = await fetch(imageUrl);
        if (!sourceImageRes.ok) throw new Error("Could not fetch the source image");
        
        const imageBlob = await sourceImageRes.blob();
        
        const formData = new FormData();
        formData.append('files', imageBlob, 'image.jpg');

        const uploadRes = await fetch(`${baseUrl}/upload`, { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();
        
        let serverFilePath;
        if (Array.isArray(uploadData)) {
            serverFilePath = typeof uploadData[0] === 'string' ? uploadData[0] : (uploadData[0].name || uploadData[0].path);
        } else {
            serverFilePath = uploadData.name || uploadData.path || uploadData;
        }

        if (!serverFilePath || serverFilePath === 'undefined') return null;

        await fetch(`${baseUrl}/queue/join`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [{ path: serverFilePath }], fn_index: 0, session_hash: sessionHash })
        });

        const eventRes = await fetch(`${baseUrl}/queue/data?session_hash=${sessionHash}`);
        const reader = eventRes.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let outputImageUrl = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const events = decoder.decode(value).split('\n\n'); 
            for (const event of events) {
                if (event.startsWith('data: ')) {
                    try {
                        const parsed = JSON.parse(event.replace('data: ', ''));
                        if (parsed.msg === 'process_completed' && parsed.success) {
                            const outData = parsed.output.data[0];
                            outputImageUrl = typeof outData === 'string' ? outData : (outData.url || `${baseUrl}/file=${outData.path || outData.name}`);
                            if (!outputImageUrl.startsWith('http')) outputImageUrl = `${baseUrl}${outputImageUrl}`;
                            break;
                        }
                    } catch (e) {}
                }
            }
            if (outputImageUrl) break;
        }

        if (outputImageUrl) {
            const imgRes = await fetch(outputImageUrl);
            const arrayBuffer = await imgRes.arrayBuffer();
            
            const fileName = `result-${Date.now()}.png`;
            const savePath = path.join(__dirname, 'rmbg', fileName);
            
            fs.writeFileSync(savePath, Buffer.from(arrayBuffer));
            
            setTimeout(() => {
                fs.unlink(savePath, (err) => {});
            }, 60000);
            
            return `${hostUrl}/rmbg/${fileName}`;
        }

    } catch (error) {
        return null;
    }
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 API is running on http://localhost:${PORT}`);
});
