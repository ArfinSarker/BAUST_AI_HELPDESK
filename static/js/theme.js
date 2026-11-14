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

  // Purge legacy shared key to prevent cross-portal theme bleed
  try {
    localStorage.removeItem('baust_theme');
  } catch (e) {}

  /**
   * Resolve storage key strictly based on active site scope.
   * Student site: 'baust_student_theme'
   * Admin site:   'baust_admin_theme'
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

  const STORAGE_KEY = getStorageKey();

  /**
   * Retrieves the current effective theme.
   * Scoped strictly to the active portal (baust_student_theme vs baust_admin_theme).
   * Default: 'dark' (Night Mode)
   */
  function getPreferredTheme() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === 'dark' || stored === 'light') return stored;
    } catch (e) {
      console.warn('localStorage access failed:', e);
    }
    return 'dark';
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

  // Attach event listener once DOM is ready
  document.addEventListener('DOMContentLoaded', function () {
    applyTheme(document.documentElement.getAttribute('data-theme') || getPreferredTheme());

    const toggleBtn = document.getElementById('themeToggleBtn');
    if (toggleBtn) {
      toggleBtn.onclick = toggleTheme;
    }
  });
})();

