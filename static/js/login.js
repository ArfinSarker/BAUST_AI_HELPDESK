/**
 * BAUST Knowledge Portal - Administrator Login Engine
 * Features:
 * 1. Strict CSP Compliant (Zero inline event handlers)
 * 2. Double-Submit Prevention with Loading State
 * 3. Alert Box Auto-Dismiss with Smooth Fade Out
 * 4. Password Visibility Toggle
 * 5. Dynamic Theme Controller (Day / Night Mode)
 */

/* Theme Management */
function syncThemeUI(theme) {
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");

  const SUN_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';
  const MOON_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
  const EYE_OPEN_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
  const EYE_OFF_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>';

  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    if (themeIcon) themeIcon.innerHTML = SUN_SVG;
    if (themeLabel) themeLabel.textContent = "Day";
    localStorage.setItem("baust_admin_theme", "dark");
  } else {
    document.documentElement.setAttribute("data-theme", "light");
    if (themeIcon) themeIcon.innerHTML = MOON_SVG;
    if (themeLabel) themeLabel.textContent = "Night";
    localStorage.setItem("baust_admin_theme", "light");
  }
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  const nextTheme = (currentTheme === "dark") ? "light" : "dark";
  syncThemeUI(nextTheme);
}

/* Password Toggle Visibility */
function togglePasswordVisibility() {
  const passwordInput = document.getElementById("password");
  const eyeIcon = document.getElementById("eyeIcon");
  if (!passwordInput) return;

  if (passwordInput.type === "password") {
    passwordInput.type = "text";
    if (eyeIcon) eyeIcon.innerHTML = EYE_OFF_SVG;
  } else {
    passwordInput.type = "password";
    if (eyeIcon) eyeIcon.innerHTML = EYE_OPEN_SVG;
  }
}

/* Event Binding & DOM Initialization (CSP Compliant) */
document.addEventListener("DOMContentLoaded", function() {
  // 1. Initial Theme Sync
  const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
  syncThemeUI(activeTheme);

  // 2. Theme Toggle Button Listener
  const themeBtn = document.getElementById("themeToggleBtn");
  if (themeBtn) {
    themeBtn.addEventListener("click", toggleTheme);
  }

  // 3. Password Visibility Button Listener
  const togglePassBtn = document.getElementById("togglePasswordBtn") || document.querySelector(".toggle-password-btn");
  if (togglePassBtn) {
    togglePassBtn.addEventListener("click", togglePasswordVisibility);
  }

  // 4. Double Submit Prevention on Login Form
  const loginForm = document.getElementById("loginForm");
  const submitBtn = document.getElementById("loginSubmitBtn");
  if (loginForm && submitBtn) {
    loginForm.addEventListener("submit", function() {
      submitBtn.disabled = true;
      submitBtn.style.opacity = "0.75";
      submitBtn.style.cursor = "not-allowed";
      submitBtn.innerHTML = '<span>⏳ Authenticating...</span>';
    });
  }

  // 5. Flash Messages & Alert Box Auto-Dismiss (Fade Out after 5.5s)
  const alerts = document.querySelectorAll(".alert-box");
  if (alerts.length > 0) {
    setTimeout(function() {
      alerts.forEach(function(alert) {
        alert.style.transition = "opacity 0.6s ease, transform 0.6s ease";
        alert.style.opacity = "0";
        alert.style.transform = "translateY(-8px)";
        setTimeout(function() {
          alert.remove();
        }, 600);
      });
    }, 5500);
  }
});

