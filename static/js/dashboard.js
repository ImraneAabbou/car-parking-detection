/**
 * ParkVision Dashboard JavaScript
 * Handles live polling, UI updates, and animations.
 */

// --- Tab Switching Logic ---
window.switchTab = function(tabId) {
    // Buttons
    const btnVideo = document.getElementById('tab-btn-video');
    const btnMap = document.getElementById('tab-btn-map');
    
    // Contents
    const contentVideo = document.getElementById('tab-content-video');
    const contentMap = document.getElementById('tab-content-map');

    if (!btnVideo || !btnMap || !contentVideo || !contentMap) return;

    // Reset Buttons
    [btnVideo, btnMap].forEach(btn => {
        btn.classList.remove('active', 'text-white', 'bg-white/10', 'shadow-sm');
        btn.classList.add('text-surface-200/60');
    });

    // Reset Contents
    [contentVideo, contentMap].forEach(content => {
        content.classList.add('hidden', 'opacity-0');
    });

    if (tabId === 'video') {
        btnVideo.classList.add('active', 'text-white', 'bg-white/10', 'shadow-sm');
        btnVideo.classList.remove('text-surface-200/60');
        
        contentVideo.classList.remove('hidden');
        // Slight delay for smooth fade
        setTimeout(() => contentVideo.classList.remove('opacity-0'), 10);
    } else if (tabId === 'map') {
        btnMap.classList.add('active', 'text-white', 'bg-white/10', 'shadow-sm');
        btnMap.classList.remove('text-surface-200/60');
        
        contentMap.classList.remove('hidden');
        // Slight delay for smooth fade
        setTimeout(() => contentMap.classList.remove('opacity-0'), 10);
    }
};

document.addEventListener("DOMContentLoaded", () => {
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
        const now = new Date();
        clock.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // --- Utility: Video Load ---
    if (videoFeed.complete) {
        document.querySelector('.glass-card').classList.add('video-loaded');
    } else {
        videoFeed.addEventListener('load', () => {
            document.querySelector('.glass-card').classList.add('video-loaded');
        });
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
            timeSpan.className = "text-[10px] text-surface-200/50 font-mono block mb-0.5";
            timeSpan.textContent = ev.time;
            
            const msgSpan = document.createElement("span");
            msgSpan.className = "text-surface-200/90";
            msgSpan.textContent = ev.message;
            
            content.appendChild(timeSpan);
            content.appendChild(msgSpan);
            
            item.appendChild(dot);
            item.appendChild(content);
            
            eventList.appendChild(item);
        });
    }

    // --- Start Polling ---
    // Fetch status every 2 seconds
    setInterval(fetchStatus, 2000);
    // Fetch events every 3 seconds
    setInterval(fetchEvents, 3000);
    
    // Initial fetch
    fetchStatus();
    fetchEvents();
});
