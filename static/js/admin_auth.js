/**
 * ============================================================================
 * BAUST Knowledge Portal - Administrator Authentication Logic (admin_auth.js)
 * 
 * Target: templates/admin_login.html
 * Features:
 * - Password visibility toggle (👁️ / 🙈)
 * - Double-submit prevention with loading indicator ("Processing...")
 * - Auto-dismissing alerts & flash messages with smooth fade-out
 * - Zero inline event listeners (Strict CSP compliant)
 * ============================================================================
 */

document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  // 1. Password Visibility Toggle
  const togglePassBtn = document.getElementById('togglePasswordBtn') || document.querySelector('.toggle-password-btn');
  const passwordInput = document.getElementById('password');
  const eyeIcon = document.getElementById('eyeIcon');

  if (togglePassBtn && passwordInput) {
    togglePassBtn.addEventListener('click', function () {
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';
      if (eyeIcon) {
        eyeIcon.textContent = isPassword ? '🙈' : '👁️';
      }
      togglePassBtn.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
    });
  }

  // 2. Double-Submit Prevention & Loading State
  const loginForm = document.getElementById('loginForm');
  const submitBtn = document.getElementById('loginSubmitBtn');

  if (loginForm && submitBtn) {
    loginForm.addEventListener('submit', function (e) {
      if (submitBtn.dataset.submitting === 'true') {
        e.preventDefault();
        return;
      }
      submitBtn.dataset.submitting = 'true';
      submitBtn.disabled = true;
      submitBtn.style.opacity = '0.75';
      submitBtn.style.cursor = 'not-allowed';
      submitBtn.innerHTML = '<span>⏳ Processing & Verifying...</span>';
    });
  }

  // 3. Flash Messages & Alert Box Auto-Dismiss (Fade Out after 5s)
  const alerts = document.querySelectorAll('.alert-box');
  if (alerts.length > 0) {
    setTimeout(function () {
      alerts.forEach(function (alert) {
        alert.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        alert.style.opacity = '0';
        alert.style.transform = 'translateY(-8px)';
        setTimeout(function () {
          alert.remove();
        }, 500);
      });
    }, 5000);
  }

  // 4. Autofocus on username input if empty
  const usernameInput = document.getElementById('username');
  if (usernameInput && !usernameInput.value) {
    usernameInput.focus();
  }
});

