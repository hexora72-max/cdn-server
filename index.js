const axios = require("axios");

const { TelegramClient } = require("teleproto");
const { StringSession } = require("teleproto/sessions");

const API_ID = 33449968;
const API_HASH = "497604288ff6e97fdd82b96c87bc74cf";

const BOT_TOKEN = "8952339875:AAG6D-hRqFG9GCtpGfJi6sfFDmwTcwoa1BE";

// Upload destination
const CHAT_ID = 6879455911;

const VIDEO_URL =
    "https://hgasm3.com/Nikuen%201%20Subbed.mp4";

const FILE_NAME = "Nikuen 1 Subbed.mp4";

const client = new TelegramClient(
    new StringSession(""),
    API_ID,
    API_HASH,
    {
        connectionRetries: 5
    }
);


// =====================================
// GET REMOTE FILE STREAM
// =====================================

async function getVideoStream() {

    console.log("🌐 Connecting to source...");

    const response = await axios.get(
        VIDEO_URL,
        {
            responseType: "stream",
            timeout: 0,
            maxContentLength: Infinity,
            maxBodyLength: Infinity,

            headers: {
                "User-Agent":
                    "Mozilla/5.0"
            }
        }
    );

    const size =
        Number(
            response.headers["content-length"]
        ) || 0;

    if (size) {
        console.log(
            "📦 Size:",
            (size / 1024 / 1024).toFixed(2),
            "MB"
        );
    } else {
        console.log(
            "📦 Size: unknown"
        );
    }

    return {
        stream: response.data,
        size
    };
}


// =====================================
// UPLOAD STREAM TO TELEGRAM
// =====================================

async function uploadStream() {

    console.log(
        "📤 Starting Telegram upload..."
    );

    const {
        stream,
        size
    } = await getVideoStream();

    let uploaded = 0;

    stream.on("data", chunk => {

        uploaded += chunk.length;

        process.stdout.write(
            `\r📤 ${(uploaded / 1024 / 1024).toFixed(2)} MB`
        );
    });


    /*
     * IMPORTANT:
     *
     * This requires a Teleproto version
     * that accepts a readable stream as
     * an upload source.
     */

    const message =
        await client.sendFile(
            CHAT_ID,
            {
                file: stream,

                caption:
                    "Nikuen 1 Subbed",

                supportsStreaming:
                    true,

                forceDocument:
                    false,

                workers: 4
            }
        );


    console.log(
        "\n\n✅ Telegram upload completed!"
    );

    console.log(
        "Message ID:",
        message.id
    );

    return message;
}


// =====================================
// MAIN
// =====================================

async function main() {

    if (
        BOT_TOKEN ===
        "PUT_NEW_BOT_TOKEN_HERE"
    ) {

        throw new Error(
            "❌ New BotFather token එක දාන්න."
        );
    }


    console.log(
        "🔌 Connecting to Telegram..."
    );


    await client.start({
        botAuthToken:
            BOT_TOKEN
    });


    console.log(
        "✅ Telegram connected!"
    );


    const message =
        await uploadStream();


    console.log(
        "\n=============================="
    );

    console.log(
        "🎉 DONE"
    );

    console.log(
        "=============================="
    );

    console.log(
        "Message ID:",
        message.id
    );
}


main().catch(error => {

    console.error(
        "\n❌ ERROR:"
    );

    console.error(
        error?.stack ||
        error?.message ||
        error
    );

});
