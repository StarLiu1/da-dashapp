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
