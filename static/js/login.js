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

  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
    if (themeIcon) themeIcon.textContent = "☀️";
    if (themeLabel) themeLabel.textContent = "Day";
    localStorage.setItem("baust_admin_theme", "dark");
  } else {
    document.documentElement.setAttribute("data-theme", "light");
    if (themeIcon) themeIcon.textContent = "🌙";
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
    if (eyeIcon) eyeIcon.textContent = "🙈";
  } else {
    passwordInput.type = "password";
    if (eyeIcon) eyeIcon.textContent = "👁️";
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

