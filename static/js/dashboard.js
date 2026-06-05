(function initTheme() {
    const saved = localStorage.getItem('parkvision-theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
    }
})();

window.toggleTheme = function () {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('parkvision-theme', next);
    updateThemeIcons(next);
};

function updateThemeIcons(theme) {
    const darkIcon = document.getElementById('theme-icon-dark');
    const lightIcon = document.getElementById('theme-icon-light');
    if (!darkIcon || !lightIcon) return;
    if (theme === 'dark') {
        darkIcon.classList.remove('hidden');
        lightIcon.classList.add('hidden');
    } else {
        darkIcon.classList.add('hidden');
        lightIcon.classList.remove('hidden');
    }
}

window.switchTab = function (tabId) {
    const btnVideo = document.getElementById('tab-btn-video');
    const btnMap = document.getElementById('tab-btn-map');
    const btnAccidents = document.getElementById('tab-btn-accidents');
    const contentVideo = document.getElementById('tab-content-video');
    const contentMap = document.getElementById('tab-content-map');
    const contentAccidents = document.getElementById('tab-content-accidents');

    if (!btnVideo || !btnMap || !contentVideo || !contentMap) return;

    [btnVideo, btnMap, btnAccidents].forEach(btn => {
        if (btn) btn.classList.remove('active');
    });

    [contentVideo, contentMap, contentAccidents].forEach(content => {
        if (content) content.classList.add('hidden', 'opacity-0');
    });

    if (tabId === 'video') {
        btnVideo.classList.add('active');
        contentVideo.classList.remove('hidden');
        setTimeout(() => contentVideo.classList.remove('opacity-0'), 10);
    } else if (tabId === 'map') {
        btnMap.classList.add('active');
        contentMap.classList.remove('hidden');
        setTimeout(() => {
            contentMap.classList.remove('opacity-0');
            syncMapHeight();
        }, 10);
    } else if (tabId === 'accidents') {
        btnAccidents.classList.add('active');
        contentAccidents.classList.remove('hidden');
        setTimeout(() => contentAccidents.classList.remove('opacity-0'), 10);
    }
};

function syncMapHeight() {
    const mapTab = document.getElementById('tab-content-map');
    const mapBody = document.getElementById('parking-map-body');
    const mainRow = document.getElementById('main-row');

    if (!mapTab || !mapBody || !mainRow) return;

    const availHeight = mainRow.offsetHeight || window.innerHeight - 250;
    const targetHeight = Math.max(availHeight, 500);

    mapTab.style.height = targetHeight + 'px';
    const headerEl = mapTab.querySelector('.section-border');
    const headerHeight = headerEl ? headerEl.offsetHeight : 48;
    mapBody.style.height = (targetHeight - headerHeight) + 'px';
    mapBody.style.maxHeight = (targetHeight - headerHeight) + 'px';

    requestAnimationFrame(() => resizeCanvas());
}

// ─── Canvas Mini-Map ──────────────────────────────────────────────
const canvas = document.getElementById('parking-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;

let canvasScale = 1;

function resizeCanvas() {
    if (!canvas || !ctx) return;

    const wrapper = document.getElementById('parking-canvas-wrapper');
    if (!wrapper) return;

    const rect = wrapper.getBoundingClientRect();
    const availW = rect.width;
    const availH = rect.height;

    if (availW <= 0 || availH <= 0) return;

    const frameRatio = VIDEO_FRAME_WIDTH / VIDEO_FRAME_HEIGHT;
    let drawW, drawH;

    if (availW / availH > frameRatio) {
        drawH = availH;
        drawW = availH * frameRatio;
    } else {
        drawW = availW;
        drawH = availW / frameRatio;
    }

    canvas.width = Math.round(drawW * window.devicePixelRatio || 1);
    canvas.height = Math.round(drawH * window.devicePixelRatio || 1);
    canvas.style.width = Math.round(drawW) + 'px';
    canvas.style.height = Math.round(drawH) + 'px';

    canvasScale = canvas.width / VIDEO_FRAME_WIDTH;

    drawCanvas();
}

function drawCanvas() {
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (_lastSpots) {
        renderSpotsOnCanvas(_lastSpots);
    }
}

let _lastSpots = null;

function renderSpotsOnCanvas(spots) {
    if (!ctx) return;

    const sx = SPOT_WIDTH * canvasScale;
    const sy = SPOT_HEIGHT * canvasScale;
    // Enforce min 2px so spots are always visible
    const sw = Math.max(2, sx);
    const sh = Math.max(2, sy);

    spots.forEach(spot => {
        const pos = SPOT_POSITIONS[spot.id - 1];
        if (!pos) return;

        const cx = pos.x * canvasScale;
        const cy = pos.y * canvasScale;

        if (spot.status === 'available') {
            ctx.fillStyle = 'rgba(16, 185, 129, 0.65)';
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
        } else {
            ctx.fillStyle = 'rgba(244, 63, 94, 0.65)';
            ctx.strokeStyle = 'rgba(244, 63, 94, 0.9)';
        }

        ctx.fillRect(cx, cy, sw, sh);
        ctx.lineWidth = 1;
        ctx.strokeRect(cx, cy, sw, sh);

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `bold ${Math.max(5, Math.round(sw * 0.5))}px sans-serif`;
        ctx.fillText(spot.id, cx + sw / 2, cy + sh / 2);
    });
}

// ─── Main Dashboard Logic ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    updateThemeIcons(currentTheme);

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }

    const statOccupied = document.getElementById("stat-occupied");
    const statAvailable = document.getElementById("stat-available");
    const statPct = document.getElementById("stat-pct");
    const statPctBar = document.getElementById("stat-pct-bar");
    const eventList = document.getElementById("event-list");
    const eventCount = document.getElementById("event-count");
    const videoFeed = document.getElementById("video-feed");
    const accidentFeed = document.getElementById("accident-feed");
    const clock = document.getElementById("clock");

    let lastKnownEventsCount = 0;

    // --- Canvas init ---
    resizeCanvas();

    function updateClock() {
        clock.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    setInterval(updateClock, 1000);
    updateClock();

    const videoParent = videoFeed ? videoFeed.closest('.glass-card') : null;
    if (videoFeed && videoParent) {
        if (videoFeed.complete) {
            videoParent.classList.add('video-loaded');
        } else {
            videoFeed.addEventListener('load', () => {
                videoParent.classList.add('video-loaded');
            });
        }
    }

    const accidentParent = accidentFeed ? accidentFeed.closest('.glass-card') : null;
    if (accidentFeed && accidentParent) {
        if (accidentFeed.complete) {
            accidentParent.classList.add('video-loaded');
        } else {
            accidentFeed.addEventListener('load', () => {
                accidentParent.classList.add('video-loaded');
            });
        }
    }

    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            statOccupied.textContent = data.occupied;
            statAvailable.textContent = data.available;
            statPct.textContent = data.occupancy_pct;
            statPctBar.style.width = `${data.occupancy_pct}%`;

            _lastSpots = data.spots;
            drawCanvas();
        } catch (error) {
            console.error("Failed to fetch parking status:", error);
        }
    }

    async function fetchEvents() {
        try {
            const res = await fetch("/api/events");
            const data = await res.json();

            const events = data.events;

            if (events.length > 0 && events.length !== lastKnownEventsCount) {
                lastKnownEventsCount = events.length;
                renderEvents(events);
            }
        } catch (error) {
            console.error("Failed to fetch events:", error);
        }
    }

    function renderEvents(events) {
        eventCount.textContent = events.length;

        eventList.innerHTML = "";

        events.forEach(ev => {
            const item = document.createElement("div");
            item.className = "event-item text-sm";

            const dot = document.createElement("div");
            dot.className = `event-dot ${ev.type}`;

            const content = document.createElement("div");

            const timeSpan = document.createElement("span");
            timeSpan.className = "text-[10px] event-time font-mono block mb-0.5";
            timeSpan.textContent = ev.time;

            const msgSpan = document.createElement("span");
            msgSpan.className = "event-msg";
            msgSpan.textContent = ev.message;

            content.appendChild(timeSpan);
            content.appendChild(msgSpan);

            item.appendChild(dot);
            item.appendChild(content);

            eventList.appendChild(item);
        });
    }

    window.addEventListener('resize', () => {
        const mapTab = document.getElementById('tab-content-map');
        if (mapTab && !mapTab.classList.contains('hidden')) {
            syncMapHeight();
        }
    });

    // Re-draw canvas whenever map becomes visible
    const mapObserver = new MutationObserver(() => {
        const mapTab = document.getElementById('tab-content-map');
        if (mapTab && !mapTab.classList.contains('hidden')) {
            requestAnimationFrame(() => resizeCanvas());
        }
    });
    const mapTab = document.getElementById('tab-content-map');
    if (mapTab) {
        mapObserver.observe(mapTab, { attributes: true, attributeFilter: ['class'] });
    }

    setInterval(fetchStatus, 2000);
    setInterval(fetchEvents, 3000);

    fetchStatus();
    fetchEvents();
});
