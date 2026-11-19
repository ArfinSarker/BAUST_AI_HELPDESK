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

  const EYE_OPEN_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>';
  const EYE_OFF_SVG = '<svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>';

  // 1. Password Visibility Toggle
  const togglePassBtn = document.getElementById('togglePasswordBtn') || document.querySelector('.toggle-password-btn');
  const passwordInput = document.getElementById('password');
  const eyeIcon = document.getElementById('eyeIcon');

  if (togglePassBtn && passwordInput) {
    togglePassBtn.addEventListener('click', function () {
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';
      if (eyeIcon) {
        eyeIcon.innerHTML = isPassword ? EYE_OFF_SVG : EYE_OPEN_SVG;
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
      submitBtn.innerHTML = '<span style="display:inline-flex;align-items:center;gap:6px;"><svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="animation: spin 1s linear infinite;"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg><span>Verifying Credentials...</span></span>';
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

