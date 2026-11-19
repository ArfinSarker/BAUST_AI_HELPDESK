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

  const SUN_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
  const MOON_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

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
      themeIcon.innerHTML = isDark ? SUN_SVG : MOON_SVG;
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

