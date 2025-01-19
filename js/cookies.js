// Cookie utility functions
const Cookies = {
    // Set a cookie with optional expiry and path
    set: function(name, value, days = 7, path = '/') {
        const expires = new Date(Date.now() + days * 864e5).toUTCString();
        document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=${path}; Secure; SameSite=Strict`;
    },

    // Get a cookie value by name
    get: function(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return decodeURIComponent(parts.pop().split(';').shift());
        }
        return null;
    },

    // Remove a cookie
    remove: function(name, path = '/') {
        document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=${path}; Secure; SameSite=Strict`;
    },

    // Check if a cookie exists
    exists: function(name) {
        return this.get(name) !== null;
    }
};

// Export the Cookies object
window.Cookies = Cookies;
