/**
 * ============================================================================
 * BAUST Knowledge Portal & AI Help Desk - Generic Theme Controller (theme.js)
 * 
 * Shared across:
 * - Student Chat Portal (user.html)
 * - Admin Dashboard (admin.html)
 * - Admin Login Gateway (admin_login.html)
 * 
 * Features:
 * - Zero-FOUC instant theme sync with localStorage & prefers-color-scheme
 * - Global window.toggleTheme() and window.applyTheme() API
 * - Automatic event listener binding on #themeToggleBtn (No inline onclick needed)
 * - Updates icons (🌙 / ☀️) and text labels (Night / Day) dynamically
 * ============================================================================
 */

(function () {
  'use strict';

  const STORAGE_KEY = 'baust_theme';

  /**
   * Retrieves the current effective theme.
   * Priority: localStorage -> OS preference -> 'light' default
   */
  function getPreferredTheme() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'dark' || stored === 'light') return stored;
    } catch (e) {
      console.warn('localStorage access failed:', e);
    }
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    return prefersDark ? 'dark' : 'light';
  }

  /**
   * Updates document data-theme and UI indicators.
   */
  function applyTheme(theme) {
    const isDark = theme === 'dark';
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');

    try {
      localStorage.setItem(STORAGE_KEY, isDark ? 'dark' : 'light');
    } catch (e) {}

    // Update UI Elements if present on the page
    const themeIcon = document.getElementById('themeIcon');
    if (themeIcon) {
      themeIcon.textContent = isDark ? '☀️' : '🌙';
    }

    const themeLabel = document.getElementById('themeLabel');
    if (themeLabel) {
      themeLabel.textContent = isDark ? 'Day' : 'Night';
    }

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn) {
      toggleBtn.setAttribute('aria-label', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
      toggleBtn.setAttribute('title', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
  }

  /**
   * Toggles between dark and light themes.
   */
  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
    const nextTheme = current === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  // Export to global window scope
  window.applyTheme = applyTheme;
  window.toggleTheme = toggleTheme;
  window.getPreferredTheme = getPreferredTheme;

  // Immediately apply preferred theme
  applyTheme(getPreferredTheme());

  // Attach event listener once DOM is ready (eliminates inline onclick)
  document.addEventListener('DOMContentLoaded', function () {
    // Re-sync UI in case elements weren't in DOM during immediate execution
    applyTheme(document.documentElement.getAttribute('data-theme') || getPreferredTheme());

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleTheme);
    }

    // Listen for OS system theme changes if user hasn't explicitly set a preference
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
        if (!localStorage.getItem(STORAGE_KEY)) {
          applyTheme(e.matches ? 'dark' : 'light');
        }
      });
    }
  });
})();

