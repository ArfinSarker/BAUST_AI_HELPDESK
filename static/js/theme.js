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

  /**
   * Resolve storage key based on active site scope.
   * Student site: 'baust_student_theme'
   * Admin site: 'baust_admin_theme'
   */
  function getStorageKey() {
    try {
      const explicitScope = document.documentElement.getAttribute('data-theme-scope');
      if (explicitScope === 'admin') return 'baust_admin_theme';
      if (explicitScope === 'student') return 'baust_student_theme';

      const path = (window.location && window.location.pathname) ? window.location.pathname : '';
      if (path.indexOf('/admin') !== -1) return 'baust_admin_theme';
    } catch (e) {}
    return 'baust_student_theme';
  }

  /**
   * Retrieves the current effective theme.
   * Priority: scoped localStorage -> legacy baust_theme fallback -> OS preference -> 'light' default
   */
  function getPreferredTheme() {
    const key = getStorageKey();
    try {
      const stored = localStorage.getItem(key);
      if (stored === 'dark' || stored === 'light') return stored;

      // Fallback for first load if legacy unified key existed
      const legacy = localStorage.getItem('baust_theme');
      if (legacy === 'dark' || legacy === 'light') return legacy;
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
      localStorage.setItem(getStorageKey(), isDark ? 'dark' : 'light');
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
        if (!localStorage.getItem(getStorageKey())) {
          applyTheme(e.matches ? 'dark' : 'light');
        }
      });
    }
  });
})();

