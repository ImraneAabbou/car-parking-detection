/**
 * ParkVision Dashboard JavaScript
 * Handles live polling, UI updates, theme switching, and animations.
 */

// ─── Theme Toggle Logic ────────────────────────────────────────────
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


// ─── Tab Switching Logic ───────────────────────────────────────────
window.switchTab = function (tabId) {
    const btnVideo = document.getElementById('tab-btn-video');
    const btnMap = document.getElementById('tab-btn-map');
    const contentVideo = document.getElementById('tab-content-video');
    const contentMap = document.getElementById('tab-content-map');

    if (!btnVideo || !btnMap || !contentVideo || !contentMap) return;

    // Reset all tabs
    [btnVideo, btnMap].forEach(btn => {
        btn.classList.remove('active');
    });

    [contentVideo, contentMap].forEach(content => {
        content.classList.add('hidden', 'opacity-0');
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
    }
};


// ─── Sync Parking Map height to match video tab ────────────────────
function syncMapHeight() {
    const videoTab = document.getElementById('tab-content-video');
    const mapTab = document.getElementById('tab-content-map');
    const mapBody = document.getElementById('parking-map-body');

    if (!videoTab || !mapTab || !mapBody) return;

    // Get the video tab's total height (it may be hidden, so measure first)
    const videoIsHidden = videoTab.classList.contains('hidden');

    // Temporarily show video tab to measure
    if (videoIsHidden) {
        videoTab.style.visibility = 'hidden';
        videoTab.style.position = 'absolute';
        videoTab.classList.remove('hidden');
    }

    const videoHeight = videoTab.offsetHeight;

    // Restore hidden state
    if (videoIsHidden) {
        videoTab.classList.add('hidden');
        videoTab.style.visibility = '';
        videoTab.style.position = '';
    }

    // Set map tab to same total height
    if (videoHeight > 0) {
        mapTab.style.height = videoHeight + 'px';
        // Calculate body height = total - header
        const headerEl = mapTab.querySelector('.section-border');
        const headerHeight = headerEl ? headerEl.offsetHeight : 48;
        mapBody.style.height = (videoHeight - headerHeight) + 'px';
        mapBody.style.maxHeight = (videoHeight - headerHeight) + 'px';
    }
}


// ─── Main Dashboard Logic ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Initialize theme icons
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    updateThemeIcons(currentTheme);

    // Bind theme toggle button
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }

    // DOM Elements
    const statOccupied = document.getElementById("stat-occupied");
    const statAvailable = document.getElementById("stat-available");
    const statPct = document.getElementById("stat-pct");
    const statPctBar = document.getElementById("stat-pct-bar");
    const parkingGrid = document.getElementById("parking-grid");
    const eventList = document.getElementById("event-list");
    const eventCount = document.getElementById("event-count");
    const videoFeed = document.getElementById("video-feed");
    const clock = document.getElementById("clock");

    let lastKnownEventsCount = 0;

    // --- Utility: Clock ---
    function updateClock() {
        clock.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // --- Utility: Video Load ---
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

    // --- Poll: API Status ---
    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();

            // Update Stats
            statOccupied.textContent = data.occupied;
            statAvailable.textContent = data.available;
            statPct.textContent = data.occupancy_pct;
            statPctBar.style.width = `${data.occupancy_pct}%`;

            // Update Parking Grid
            updateParkingGrid(data.spots);
        } catch (error) {
            console.error("Failed to fetch parking status:", error);
        }
    }

    // --- Update Parking Grid ---
    function updateParkingGrid(spots) {
        if (parkingGrid.children.length === 0) {
            // First time initialization
            spots.forEach(spot => {
                const div = document.createElement("div");
                div.id = `spot-${spot.id}`;
                div.textContent = spot.id;
                div.className = `spot-cell ${spot.status === 'available' ? 'spot-available' : 'spot-occupied'}`;
                parkingGrid.appendChild(div);
            });
        } else {
            // Update existing
            spots.forEach(spot => {
                const div = document.getElementById(`spot-${spot.id}`);
                if (div) {
                    const newClass = spot.status === 'available' ? 'spot-available' : 'spot-occupied';
                    if (!div.classList.contains(newClass)) {
                        div.className = `spot-cell ${newClass}`;
                    }
                }
            });
        }
    }

    // --- Poll: API Events ---
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

    // --- Render Events ---
    function renderEvents(events) {
        eventCount.textContent = events.length;

        // Clear list
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

    // --- Sync map height on window resize ---
    window.addEventListener('resize', () => {
        const mapTab = document.getElementById('tab-content-map');
        if (mapTab && !mapTab.classList.contains('hidden')) {
            syncMapHeight();
        }
    });

    // --- Start Polling ---
    setInterval(fetchStatus, 2000);
    setInterval(fetchEvents, 3000);

    // Initial fetch
    fetchStatus();
    fetchEvents();
});
