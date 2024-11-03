// assets/tooltip.js
document.addEventListener('DOMContentLoaded', function () {
    const tooltipTrigger = document.querySelector('.question-mark');
    const tooltip = document.querySelector('.tooltip');

    tooltipTrigger.addEventListener('mouseenter', function (event) {
        const rect = tooltipTrigger.getBoundingClientRect();
        
        // Adjust the tooltip position relative to the question mark
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`;  // Place it just above the question mark
        
        // Show the tooltip
        tooltip.style.display = 'block';
    });

    tooltipTrigger.addEventListener('mouseleave', function () {
        // Hide the tooltip when not hovering
        tooltip.style.display = 'none';
    });
});

// assets/tooltip.js
document.addEventListener("DOMContentLoaded", function () {
    console.log("Tooltip JavaScript loaded");

    const observer = new MutationObserver(() => {
        const overlay = document.getElementById("roc-loading-overlay");
        if (overlay) {
            console.log("Overlay found, adding click listener");
            overlay.addEventListener("click", function () {
                console.log("Overlay clicked");
                overlay.style.display = "none";
            });
            observer.disconnect();  // Stop observing once the listener is added
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
});

document.addEventListener("DOMContentLoaded", function () {
    console.log("Tooltip JavaScript loaded");

    const observer = new MutationObserver(() => {
        const overlay = document.getElementById("apar-loading-overlay");
        if (overlay) {
            console.log("Overlay found, adding click listener");
            overlay.addEventListener("click", function () {
                console.log("Overlay clicked");
                overlay.style.display = "none";
            });
            observer.disconnect();  // Stop observing once the listener is added
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
});


