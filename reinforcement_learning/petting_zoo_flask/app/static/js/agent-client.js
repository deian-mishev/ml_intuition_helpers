const vid = document.getElementById('video');
const pressedKeys = new Set();
const title = document.getElementById('title');
const envSelect = document.getElementById('envSelect');
const playerSelect = document.getElementById('playerSelect');
const startBtn = document.getElementById('startBtn');
const screen_container = document.getElementById('screen-container');
let isConnected = false;
let socket = null;
let lastSendTime = 0;
const SEND_INTERVAL_MS = 50;

function handleKeydown(e) {
    pressedKeys.add(e.key);
}

function handleKeyup(e) {
    pressedKeys.delete(e.key);
}

function checkEnableStart() {
    startBtn.disabled = !(envSelect.value && playerSelect.value && !isConnected);
}
function gameLoop(timestamp) {
    if (timestamp - lastSendTime >= SEND_INTERVAL_MS) {
        if (pressedKeys.size > 0) {
            const payload = Array.from(pressedKeys);
            socket.emit('input', payload);
        }
        lastSendTime = timestamp;
    }

    animationFrameId = requestAnimationFrame(gameLoop);
}

function cleanup() {
    if (!isConnected) return;
    isConnected = false;

    document.removeEventListener("keydown", handleKeydown);
    document.removeEventListener("keyup", handleKeyup);
    cancelAnimationFrame(animationFrameId);

    if (socket?.connected) {
        socket.disconnect();
    }

    socket = null;
    title.textContent = "Episode ended, play again ..";
    screen_container.style.display = 'flex';

    checkEnableStart();
}

fetch('/preconnect')
    .then(response => response.json())
    .then(data => {
        data.environments.forEach(env => {
            const option = document.createElement("option");
            option.value = env;
            option.textContent = env;
            envSelect.appendChild(option);
        });

        data.ai_players.forEach(player => {
            const option = document.createElement("option");
            option.value = player;
            option.textContent = player;
            playerSelect.appendChild(option);
        });

        checkEnableStart();
    })
    .catch(err => {
        console.error("AI Service is down:", err);
    });

envSelect.addEventListener("change", checkEnableStart);
playerSelect.addEventListener("change", checkEnableStart);
startBtn.addEventListener("click", () => {
    if (isConnected) return;
    const environment = envSelect.value;
    const ai_player = playerSelect.value;

    if (!environment || !ai_player) return;
    socket = io({
        query: {
            env: environment,
            ai_player: ai_player
        }
    });

    socket.on("connect",
        () => {
            title.textContent = "Agent Playground";
            screen_container.style.display = 'none';
            console.log("connected");
            isConnected = true;
            checkEnableStart();

            document.addEventListener("keydown", handleKeydown);
            document.addEventListener("keyup", handleKeyup);
            requestAnimationFrame(gameLoop);
        },
        (error) => {
            console.error("Socket connection error:", error);
            cleanup();
        }
    );

    socket.on("frame", (base64Frame) => {
        vid.src = 'data:image/png;base64,' + base64Frame;
    });

    socket.on("episode_end", () => {
        cleanup();
    });

    socket.on("disconnect", () => {
        cleanup();
    });
});

window.addEventListener("beforeunload", cleanup);
