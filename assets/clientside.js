if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.clientside = {
    updateRefreshTrigger: function(pathname) {
        return Date.now();  // Return current timestamp to force a refresh
    }
};