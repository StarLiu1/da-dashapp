if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
    updateRefreshTrigger: function(pathname) {
        return Date.now();  // Return current timestamp to force a refresh
    },
    
    navigateToApar: function(n_clicks) {
        if (n_clicks) {
            // Small delay to ensure data is saved first
            setTimeout(function() {
                window.location.href = '/apar';
            }, 100);
        }
        return window.dash_clientside.no_update;
    }
};