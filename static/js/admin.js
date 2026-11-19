/**
 * ============================================================================
 * BAUST Knowledge Portal - Administrative Dashboard Engine (admin.js)
 * 
 * Target: templates/admin.html
 * Features:
 * 1. Zero inline event listeners (Pure addEventListener & delegation)
 * 2. Real-time Async Document Upload with XHR upload.onprogress
 * 3. Client-side File Size Validation (25MB limit)
 * 4. Client-side Live Search & Pagination (TablePaginator)
 * 5. Bulk Document Operations (Archive, Delete, Selection)
 * 6. Dynamic Metadata & Knowledge Gap Modals
 * 7. Loading Overlays & Form Lock States to Prevent Double Submissions
 * ============================================================================
 */

(function () {
  'use strict';

  const MAX_UPLOAD_SIZE = 25 * 1024 * 1024; // 25 MB

  // ==========================================================================
  // TOAST NOTIFICATIONS
  // ==========================================================================
  function showToast(message, type = 'info') {
    let container = document.getElementById('adminToastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'adminToastContainer';
      container.className = 'admin-toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `admin-toast ${type}`;
    let iconSvg = '';
    if (type === 'success') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    } else if (type === 'error') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#EF4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
    } else if (type === 'warning') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
    } else {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#3B82F6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    }
    toast.innerHTML = `<span style="display:inline-flex;align-items:center;">${iconSvg}</span> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(50px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // ==========================================================================
  // CUSTOM SELECT INPUT TOGGLERS
  // ==========================================================================
  function toggleCustomMetaType(selectEl) {
    if (!selectEl) return;
    const container = selectEl.closest('.form-group');
    if (!container) return;
    const customInput = container.querySelector('input[name="custom_doc_type"]') || document.getElementById('editMetaCustomType');
    if (customInput) {
      customInput.style.display = (selectEl.value === '__custom__') ? 'block' : 'none';
      if (selectEl.value === '__custom__') customInput.focus();
    }
  }

  function toggleCustomMetaAuth(selectEl) {
    if (!selectEl) return;
    const container = selectEl.closest('.form-group');
    if (!container) return;
    const customInput = container.querySelector('input[name="custom_issuing_authority"]') || document.getElementById('editMetaCustomAuth');
    if (customInput) {
      customInput.style.display = (selectEl.value === '__custom__') ? 'block' : 'none';
      if (selectEl.value === '__custom__') customInput.focus();
    }
  }

  // ==========================================================================
  // METADATA EDIT MODAL CONTROLS
  // ==========================================================================
  function openMetaModal(fname, title, docType, ver, year, sem, auth, status, effDate, expDate) {
    const modal = document.getElementById('metaModal');
    const fInput = document.getElementById('editMetaFilename');
    const tInput = document.getElementById('editMetaTitle');
    const typeInput = document.getElementById('editMetaType');
    const verInput = document.getElementById('editMetaVersion');
    const yearInput = document.getElementById('editMetaYear');
    const semInput = document.getElementById('editMetaSemester');
    const authInput = document.getElementById('editMetaAuthority');
    const statusInput = document.getElementById('editMetaStatus');
    const effInput = document.getElementById('editMetaEffective');
    const expInput = document.getElementById('editMetaExpiry');

    if (fInput) fInput.value = fname || '';
    if (tInput) tInput.value = title || '';

    if (typeInput) {
      let hasTypeOption = Array.from(typeInput.options).some(o => o.value === docType);
      if (docType && !hasTypeOption && docType !== '__custom__') {
        const opt = document.createElement('option');
        opt.value = docType;
        opt.textContent = docType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        typeInput.insertBefore(opt, typeInput.lastElementChild);
      }
      typeInput.value = docType || 'general_notice';
      const customType = document.getElementById('editMetaCustomType');
      if (customType) customType.style.display = 'none';
    }

    if (verInput) verInput.value = ver || '1.0';
    if (yearInput) yearInput.value = year || '';
    if (semInput) semInput.value = sem || '';

    if (authInput) {
      let hasAuthOption = Array.from(authInput.options).some(o => o.value === auth);
      if (auth && !hasAuthOption && auth !== '__custom__') {
        const opt = document.createElement('option');
        opt.value = auth;
        opt.textContent = auth;
        authInput.insertBefore(opt, authInput.lastElementChild);
      }
      authInput.value = auth || 'BAUST Authority';
      const customAuth = document.getElementById('editMetaCustomAuth');
      if (customAuth) customAuth.style.display = 'none';
    }

    if (statusInput) statusInput.value = status || 'active';
    if (effInput) effInput.value = effDate || '';
    if (expInput) expInput.value = expDate || '';

    if (modal) {
      modal.classList.add('active');
      modal.style.display = 'flex';
    }
  }

  function closeMetaModal() {
    const modal = document.getElementById('metaModal');
    if (modal) {
      modal.classList.remove('active');
      modal.style.display = 'none';
    }
  }

  // ==========================================================================
  // KNOWLEDGE GAP MODAL & QUICK RESOLVE
  // ==========================================================================
  function openResolveModal(question, gapId) {
    const modal = document.getElementById('gapModal');
    const qInput = document.getElementById('modalQuestion');
    const gapInput = document.getElementById('modalGapId');
    const titleInput = document.getElementById('modalTitle');
    const contentInput = document.getElementById('modalContent');

    if (qInput) qInput.value = question || '';
    if (gapInput) gapInput.value = gapId || '';
    if (titleInput) {
      const cleanTitle = (question || 'knowledge_update')
        .replace(/[^a-zA-Z0-9\s]/g, '')
        .trim()
        .replace(/\s+/g, '_')
        .substring(0, 35);
      titleInput.value = cleanTitle;
    }
    if (contentInput) {
      contentInput.value = '';
      setTimeout(() => contentInput.focus(), 150);
    }
    if (modal) {
      modal.classList.add('active');
      modal.style.display = 'flex';
    }
  }

  function closeResolveModal() {
    const modal = document.getElementById('gapModal');
    if (modal) {
      modal.classList.remove('active');
      modal.style.display = 'none';
    }
  }

  async function markGapResolvedAjax(btn, gapId) {
    if (!gapId) return;
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span style="display:inline-block; width:11px; height:11px; border:2px solid currentColor; border-right-color:transparent; border-radius:50%; animation: spin 0.6s linear infinite; vertical-align:middle; margin-right:3px;"></span> Saving...`;
    try {
      const res = await fetch('/admin/resolve-gap/' + gapId, {
        method: 'POST'
      });
      if (res.ok) {
        const row = btn.closest('tr');
        if (row) {
          row.dataset.status = 'resolved';
          row.style.background = 'transparent';
          const iconSpan = row.querySelector('.gap-icon');
          if (iconSpan) {
            iconSpan.style.color = '#10B981';
            iconSpan.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
          }
          const statusCell = row.children[3];
          if (statusCell) {
            statusCell.innerHTML = `
              <span class="badge-status active">
                <span class="status-indicator-dot active"></span>
                <span>Resolved</span>
              </span>
            `;
          }
        }
        btn.remove();

        // Update tab badge counters dynamically
        const unresolvedBadge = document.querySelector('#gapsFilterTabs [data-filter="unresolved"] .tab-badge');
        if (unresolvedBadge) {
          let count = parseInt(unresolvedBadge.textContent, 10);
          if (!isNaN(count) && count > 0) unresolvedBadge.textContent = count - 1;
        }
        const resolvedBadge = document.querySelector('#gapsFilterTabs [data-filter="resolved"] .tab-badge');
        if (resolvedBadge) {
          let count = parseInt(resolvedBadge.textContent, 10);
          if (!isNaN(count)) resolvedBadge.textContent = count + 1;
        }

        // Re-filter gaps table if active
        if (window.gapsPaginator) {
          window.gapsPaginator.filter();
        }

        showToast('Knowledge gap marked as resolved!', 'success');
      } else {
        btn.disabled = false;
        btn.innerHTML = originalHtml;
        showToast('Failed to mark gap resolved.', 'error');
      }
    } catch (err) {
      console.error('Failed to mark resolved:', err);
      btn.disabled = false;
      btn.innerHTML = originalHtml;
      showToast('Network error.', 'error');
    }
  }

  // ==========================================================================
  // CLIENT-SIDE FILE VALIDATION, DROPZONE & PREVIEW GALLERY
  // ==========================================================================
  let stagedFiles = [];
  let stagedObjectUrls = [];

  function syncFileInputFromStaged() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput) return;
    try {
      const dt = new DataTransfer();
      stagedFiles.forEach(file => dt.items.add(file));
      fileInput.files = dt.files;
    } catch (e) {
      console.warn('DataTransfer sync warning:', e);
    }
  }

  function formatFileSize(bytes) {
    if (bytes >= 1048576) {
      return (bytes / 1048576).toFixed(2) + ' MB';
    }
    return (bytes / 1024).toFixed(1) + ' KB';
  }

  const CLOUD_UPLOAD_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>';
  const FILE_SINGLE_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>';
  const FILE_STACK_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>';

  function getFileIcon(ext) {
    if (['jpg', 'jpeg', 'png', 'webp'].includes(ext)) {
      return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>';
    }
    if (ext === 'pdf') {
      return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M10 12h4"></path><path d="M10 16h4"></path></svg>';
    }
    if (ext === 'txt') {
      return '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>';
    }
    return FILE_SINGLE_ICON;
  }

  function renderFilePreviewGallery() {
    const dropzone = document.getElementById('dropzoneBox');
    const icon = document.getElementById('dropzoneIcon');
    const title = document.getElementById('dropzoneTitle');
    const subtext = document.getElementById('dropzoneSubtext');
    const previewContainer = document.getElementById('filePreviewContainer');
    const previewCount = document.getElementById('previewCount');
    const previewGrid = document.getElementById('filePreviewGrid');
    const fileInput = document.getElementById('fileInput');

    // Revoke previous URLs to prevent memory leaks
    stagedObjectUrls.forEach(url => URL.revokeObjectURL(url));
    stagedObjectUrls = [];

    if (!stagedFiles || stagedFiles.length === 0) {
      if (dropzone) dropzone.classList.remove('has-file');
      if (icon) icon.innerHTML = CLOUD_UPLOAD_ICON;
      if (title) title.textContent = 'Click or Drag & Drop file(s) here';
      if (subtext) subtext.textContent = 'PDFs, TXT, PNG, JPG (Multiple files supported)';
      if (previewContainer) previewContainer.style.display = 'none';
      if (fileInput) fileInput.value = '';
      return;
    }

    if (dropzone) dropzone.classList.add('has-file');

    const totalBytes = stagedFiles.reduce((acc, f) => acc + f.size, 0);
    const totalSizeStr = formatFileSize(totalBytes);

    if (icon) icon.innerHTML = stagedFiles.length === 1 ? FILE_SINGLE_ICON : FILE_STACK_ICON;
    if (title) {
      title.innerHTML = `<span class="selected-filename">${stagedFiles.length} File${stagedFiles.length > 1 ? 's' : ''} Staged for AI Sync</span>`;
    }
    if (subtext) {
      subtext.innerHTML = `<span class="selected-meta">Total: ${totalSizeStr} &bull; <span style="text-decoration: underline;">Click dropzone to add more</span></span>`;
    }

    if (previewContainer) previewContainer.style.display = 'flex';
    if (previewCount) {
      previewCount.innerHTML = `<span style="display: inline-flex; align-items: center; gap: 6px;"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg> Selected Files (${stagedFiles.length})</span> <span style="font-size: 11px; font-weight: 500; color: var(--text-muted);">— ${totalSizeStr}</span>`;
    }

    if (previewGrid) {
      previewGrid.innerHTML = '';
      stagedFiles.forEach((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        const card = document.createElement('div');
        card.className = 'file-preview-card';
        card.setAttribute('data-index', index);

        // Thumbnail
        const thumb = document.createElement('div');
        thumb.className = 'file-preview-thumb';

        if (['jpg', 'jpeg', 'png', 'webp'].includes(ext)) {
          const objUrl = URL.createObjectURL(file);
          stagedObjectUrls.push(objUrl);
          const img = document.createElement('img');
          img.src = objUrl;
          img.alt = file.name;
          img.loading = 'lazy';
          thumb.appendChild(img);
        } else {
          const docIcon = document.createElement('span');
          docIcon.className = 'doc-icon';
          docIcon.textContent = getFileIcon(ext);
          thumb.appendChild(docIcon);
        }

        // Details
        const details = document.createElement('div');
        details.className = 'file-preview-details';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'file-preview-name';
        nameSpan.textContent = file.name;
        nameSpan.title = file.name;

        const sizeSpan = document.createElement('span');
        sizeSpan.className = 'file-preview-size';
        sizeSpan.textContent = formatFileSize(file.size);

        details.appendChild(nameSpan);
        details.appendChild(sizeSpan);

        // Remove button (Individual X delete)
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn-remove-file';
        removeBtn.innerHTML = '&times;';
        removeBtn.title = `Remove "${file.name}"`;
        removeBtn.setAttribute('aria-label', `Remove ${file.name}`);
        removeBtn.addEventListener('click', function (e) {
          e.preventDefault();
          e.stopPropagation();
          removeStagedFile(index);
        });

        card.appendChild(removeBtn);
        card.appendChild(thumb);
        card.appendChild(details);

        previewGrid.appendChild(card);
      });
    }
  }

  function removeStagedFile(index) {
    if (index >= 0 && index < stagedFiles.length) {
      const removed = stagedFiles.splice(index, 1)[0];
      syncFileInputFromStaged();
      renderFilePreviewGallery();
      showToast(`Removed "${removed.name}"`, 'info');
    }
  }

  function clearAllStagedFiles() {
    if (stagedFiles.length === 0) return;
    const count = stagedFiles.length;
    stagedFiles = [];
    syncFileInputFromStaged();
    renderFilePreviewGallery();
    showToast(`Cleared ${count} file(s)`, 'info');
  }

  function handleIncomingFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    let addedCount = 0;
    Array.from(fileList).forEach(file => {
      if (file.size > MAX_UPLOAD_SIZE) {
        showToast(`"${file.name}" exceeds the 25MB limit. Skipped.`, 'error');
        return;
      }

      // Avoid exact duplicates (same name and size)
      const isDuplicate = stagedFiles.some(f => f.name === file.name && f.size === file.size);
      if (!isDuplicate) {
        stagedFiles.push(file);
        addedCount++;
      }
    });

    syncFileInputFromStaged();
    renderFilePreviewGallery();

    if (addedCount > 0) {
      showToast(`Added ${addedCount} file(s) for upload.`, 'success');
    }
  }

  // ==========================================================================
  // PROGRESS OVERLAY CONTROLS & STAGED REALISTIC PROGRESS ENGINE
  // ==========================================================================
  let processingProgressInterval = null;

  function stopProgressInterval() {
    if (processingProgressInterval) {
      clearInterval(processingProgressInterval);
      processingProgressInterval = null;
    }
  }

  function setProgressStep(stepNum) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById('step' + i);
      if (el) {
        if (i < stepNum) {
          el.className = 'progress-step completed';
        } else if (i === stepNum) {
          el.className = 'progress-step active';
        } else {
          el.className = 'progress-step';
        }
      }
    }
  }

  function simulateProcessingProgress(startPct = 30) {
    stopProgressInterval();
    let current = startPct;
    setProgressStep(1);

    processingProgressInterval = setInterval(() => {
      // Gradually increases progress toward 95%, slowing down as it approaches 95%
      // 100% is STRICTLY reserved until server finishes all OCR & vector sync!
      if (current < 50) {
        current += 2;
        updateProgressBar(Math.round(current), 'Extracting text & OCR document digitization...');
        setProgressStep(1);
      } else if (current < 70) {
        current += 1.5;
        updateProgressBar(Math.round(current), 'Structuring content into clean Markdown...');
        setProgressStep(2);
      } else if (current < 85) {
        current += 1.0;
        updateProgressBar(Math.round(current), 'Categorizing topic & routing knowledge base...');
        setProgressStep(3);
      } else if (current < 95) {
        current += 0.4;
        updateProgressBar(Math.round(current), 'Generating embeddings & updating FAISS vector database...');
        setProgressStep(4);
      } else {
        updateProgressBar(95, 'Finalizing vector index synchronization...');
        setProgressStep(4);
      }
    }, 280);
  }

  function updateProgressBar(percentage, statusText) {
    const fill = document.getElementById('loadingProgressBarFill');
    const text = document.getElementById('loadingProgressBarText');
    const pctDisplay = document.getElementById('loadingProgressPercentage');

    const clamped = Math.max(0, Math.min(100, Math.round(percentage)));

    if (fill) fill.style.width = `${clamped}%`;
    if (pctDisplay) pctDisplay.textContent = `${clamped}%`;
    if (text && statusText) text.textContent = statusText;
  }

  function showLoadingOverlay(mainTitle, subTitle, showProgress = true) {
    stopProgressInterval();
    const overlay = document.getElementById('loadingOverlay');
    if (mainTitle) document.getElementById('loadingMainTitle').textContent = mainTitle;
    if (subTitle) document.getElementById('loadingSubTitle').textContent = subTitle;

    const barContainer = document.getElementById('loadingProgressBarContainer');
    if (barContainer) barContainer.style.display = 'block';

    updateProgressBar(0, 'Initiating process...');
    if (overlay) overlay.classList.add('active');
    setProgressStep(1);

    if (!showProgress) {
      // For synchronous submissions like Reindex or Direct Notice
      simulateProcessingProgress(15);
    }
  }

  function hideLoadingOverlay() {
    stopProgressInterval();
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('active');

    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = '<span id="uploadBtnIcon"><svg class="btn-svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg></span> <span id="uploadBtnText">Upload & Sync to AI Database</span>';
      uploadBtn.style.opacity = '1';
      uploadBtn.style.cursor = 'pointer';
    }

    const uploadPanel = document.getElementById('uploadPanelCard');
    const noticePanel = document.getElementById('noticePanelCard');
    if (uploadPanel) uploadPanel.classList.remove('form-locked');
    if (noticePanel) noticePanel.classList.remove('form-locked');
  }

  // ==========================================================================
  // REAL-TIME ASYNC UPLOAD (XHR onprogress)
  // ==========================================================================
  function handleUploadSubmit(event) {
    event.preventDefault();
    const form = document.getElementById('uploadForm') || event.target;

    if (!stagedFiles || stagedFiles.length === 0) {
      showToast('Please select at least one file to upload.', 'warning');
      return false;
    }

    for (let f of stagedFiles) {
      if (f.size > MAX_UPLOAD_SIZE) {
        showToast(`File "${f.name}" exceeds the 25MB limit. Please upload files under 25MB.`, 'error');
        return false;
      }
    }

    const uploadBtn = document.getElementById('uploadBtn');
    const uploadPanel = document.getElementById('uploadPanelCard');
    const noticePanel = document.getElementById('noticePanelCard');

    if (uploadBtn) {
      uploadBtn.disabled = true;
      uploadBtn.innerHTML = '<span class="btn-spinner"></span> Uploading...';
      uploadBtn.style.opacity = '0.75';
      uploadBtn.style.cursor = 'not-allowed';
    }

    if (uploadPanel) uploadPanel.classList.add('form-locked');
    if (noticePanel) noticePanel.classList.add('form-locked');

    showLoadingOverlay(
      'Uploading Document & Updating Knowledge Base...',
      'Sending document to server, extracting structured Markdown, and synchronizing vector index.',
      true
    );

    const formData = new FormData(form);
    // Guarantee formData matches stagedFiles accurately and action is explicitly set
    formData.set('action', 'upload');
    formData.delete('image');
    stagedFiles.forEach(file => {
      formData.append('image', file);
    });

    const xhr = new XMLHttpRequest();
    const targetUrl = (typeof form.getAttribute === 'function' && form.getAttribute('action')) || '/admin';
    xhr.open('POST', targetUrl, true);
    xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
    xhr.timeout = 300000; // 5-minute timeout for multi-image extraction and vector indexing

    xhr.upload.onprogress = function(e) {
      if (e.lengthComputable) {
        // Upload bytes transfer maps from 0% to 30%
        const uploadPct = Math.min(30, Math.round((e.loaded / e.total) * 30));
        updateProgressBar(uploadPct, `Uploading document (${Math.round((e.loaded / e.total) * 100)}%)...`);
        setProgressStep(1);
        if (e.loaded >= e.total) {
          // File upload finished; now smoothly load through OCR, structuring, and indexing up to 95%
          simulateProcessingProgress(32);
        }
      }
    };

    xhr.onload = function() {
      stopProgressInterval();
      if (xhr.status === 200) {
        // Now reach 100% because all server-side work is finished!
        updateProgressBar(100, 'Processing 100% complete — Vector database synchronized!');
        setProgressStep(4);
        const step4 = document.getElementById('step4');
        if (step4) step4.className = 'progress-step completed';

        showToast('Document processed & AI knowledge synchronized successfully!', 'success');
        setTimeout(() => {
          window.location.reload();
        }, 900);
      } else {
        hideLoadingOverlay();
        let errMsg = 'Upload failed with server response: ' + xhr.status;
        try {
          const resp = JSON.parse(xhr.responseText);
          if (resp && resp.message) errMsg = resp.message;
        } catch (e) {}
        showToast(errMsg, 'error');
      }
    };

    xhr.ontimeout = function() {
      stopProgressInterval();
      hideLoadingOverlay();
      showToast('Upload/processing timed out. Server may still be processing in background.', 'warning');
    };

    xhr.onerror = function() {
      stopProgressInterval();
      hideLoadingOverlay();
      showToast('Network error occurred during document upload.', 'error');
    };

    xhr.send(formData);
    return false;
  }

  function handleDirectNoticeSubmit(event) {
    const content = document.getElementById('noticeContent').value.trim();
    if (!content) {
      event.preventDefault();
      showToast('Notice content cannot be empty.', 'warning');
      return false;
    }

    const noticeBtn = document.getElementById('noticeBtn');
    const uploadPanel = document.getElementById('uploadPanelCard');
    const noticePanel = document.getElementById('noticePanelCard');

    if (noticeBtn) {
      noticeBtn.disabled = true;
      noticeBtn.innerHTML = '<span class="btn-spinner"></span> Publishing & Syncing...';
      noticeBtn.style.opacity = '0.75';
      noticeBtn.style.cursor = 'not-allowed';
    }

    if (uploadPanel) uploadPanel.classList.add('form-locked');
    if (noticePanel) noticePanel.classList.add('form-locked');

    showLoadingOverlay(
      'Publishing Notice & Indexing AI...',
      'Structuring notice into clean Markdown and syncing FAISS vector database.',
      false
    );

    return true;
  }

  function handleReindexSubmit() {
    const reindexBtn = document.getElementById('reindexBtn');
    if (reindexBtn) {
      reindexBtn.disabled = true;
      reindexBtn.innerHTML = '<span class="btn-spinner"></span> Rebuilding Vectors...';
      reindexBtn.style.opacity = '0.75';
      reindexBtn.style.cursor = 'not-allowed';
    }

    showLoadingOverlay(
      'Rebuilding All Vector Embeddings...',
      'Scanning all knowledge documents and generating multilingual cosine similarity indices.',
      false
    );

    return true;
  }

  function confirmDelete(docName) {
    const confirmed = confirm(`Are you sure you want to delete "${docName}" from the knowledge base?\nThis action will update the AI vector index automatically.`);
    if (confirmed) {
      showLoadingOverlay(
        `Deleting "${docName}"...`,
        'Removing file and re-indexing the vector database.',
        false
      );
    }
    return confirmed;
  }

  // ==========================================================================
  // CLIENT-SIDE TABLE PAGINATOR
  // ==========================================================================
  class TablePaginator {
    constructor(config) {
      this.table = document.getElementById(config.tableId);
      if (!this.table) return;

      this.tbody = this.table.querySelector('tbody');
      this.rows = Array.from(this.tbody.querySelectorAll('tr'));
      this.searchInput = document.getElementById(config.searchInputId);
      this.clearBtn = document.getElementById(config.clearBtnId);
      this.rowsSelect = document.getElementById(config.rowsSelectId);
      this.paginationNav = document.getElementById(config.paginationNavId);
      this.paginationInfo = document.getElementById(config.paginationInfoId);

      this.filteredRows = [...this.rows];
      this.currentPage = 1;
      this.rowsPerPage = parseInt(config.defaultRows || 25, 10);
      this.currentFilter = 'all';

      this.bindEvents();
      this.render(false);
    }

    scrollToTable() {
      const target = this.table.closest('details') || this.table.closest('.table-card') || this.table;
      if (target) {
        const topY = target.getBoundingClientRect().top + window.pageYOffset - 75;
        window.scrollTo({ top: Math.max(0, topY), behavior: 'smooth' });
      }
    }

    bindEvents() {
      if (this.searchInput) {
        this.searchInput.addEventListener('input', () => {
          this.filter();
          if (this.clearBtn) {
            this.clearBtn.style.display = this.searchInput.value.trim() ? 'block' : 'none';
          }
        });
      }

      if (this.clearBtn) {
        this.clearBtn.addEventListener('click', () => {
          this.searchInput.value = '';
          this.clearBtn.style.display = 'none';
          this.filter();
          this.searchInput.focus();
        });
      }

      if (this.rowsSelect) {
        this.rowsSelect.addEventListener('change', () => {
          const val = this.rowsSelect.value;
          this.rowsPerPage = val === 'all' ? 999999 : parseInt(val, 10);
          this.currentPage = 1;
          this.render(false);
        });
      }
    }

    filter() {
      const query = (this.searchInput ? this.searchInput.value : '').toLowerCase().trim();
      this.filteredRows = this.rows.filter(row => {
        if (this.currentFilter && this.currentFilter !== 'all') {
          const status = row.dataset.status;
          if (status && status !== this.currentFilter) {
            return false;
          }
        }
        if (!query) return true;
        const text = row.innerText.toLowerCase();
        return text.includes(query);
      });
      this.currentPage = 1;
      this.render(false);
    }

    render(shouldScroll = false) {
      const total = this.filteredRows.length;
      const totalPages = Math.max(1, Math.ceil(total / this.rowsPerPage));
      if (this.currentPage > totalPages) this.currentPage = totalPages;

      const startIdx = (this.currentPage - 1) * this.rowsPerPage;
      const endIdx = startIdx + this.rowsPerPage;

      // In-place visibility update to strictly prevent table height from collapsing to 0
      const activeSet = new Set(this.filteredRows.slice(startIdx, endIdx));
      this.rows.forEach(r => {
        const isVisible = activeSet.has(r);
        const targetDisplay = isVisible ? '' : 'none';
        if (r.style.display !== targetDisplay) {
          r.style.display = targetDisplay;
        }
      });

      if (this.paginationInfo) {
        if (total === 0) {
          this.paginationInfo.textContent = 'No matching records found';
        } else {
          const showStart = startIdx + 1;
          const showEnd = Math.min(endIdx, total);
          this.paginationInfo.textContent = `Showing ${showStart} to ${showEnd} of ${total} entries`;
        }
      }

      if (this.paginationNav) {
        this.paginationNav.innerHTML = '';
        if (totalPages <= 1) return;

        const prevBtn = document.createElement('button');
        prevBtn.type = 'button';
        prevBtn.className = 'btn-page';
        prevBtn.textContent = '« Prev';
        prevBtn.disabled = this.currentPage === 1;
        prevBtn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (this.currentPage > 1) {
            this.currentPage--;
            this.render(true);
          }
        });
        this.paginationNav.appendChild(prevBtn);

        let startP = Math.max(1, this.currentPage - 2);
        let endP = Math.min(totalPages, startP + 4);
        if (endP - startP < 4) startP = Math.max(1, endP - 4);

        for (let p = startP; p <= endP; p++) {
          const pageBtn = document.createElement('button');
          pageBtn.type = 'button';
          pageBtn.className = `btn-page ${p === this.currentPage ? 'active' : ''}`;
          pageBtn.textContent = p;
          pageBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (this.currentPage !== p) {
              this.currentPage = p;
              this.render(true);
            }
          });
          this.paginationNav.appendChild(pageBtn);
        }

        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'btn-page';
        nextBtn.textContent = 'Next »';
        nextBtn.disabled = this.currentPage === totalPages;
        nextBtn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (this.currentPage < totalPages) {
            this.currentPage++;
            this.render(true);
          }
        });
        this.paginationNav.appendChild(nextBtn);
      }

      if (shouldScroll) {
        this.scrollToTable();
      }
    }
  }

  // ==========================================================================
  // BULK ACTIONS ENGINE
  // ==========================================================================
  function updateBulkToolbar() {
    const checkboxes = document.querySelectorAll('.doc-checkbox:checked');
    const bar = document.getElementById('bulkActionsBar');
    const countEl = document.getElementById('bulkSelectedCount');
    const selectAll = document.getElementById('selectAllDocs');

    if (!bar || !countEl) return;

    const count = checkboxes.length;
    countEl.textContent = count;

    if (count > 0) {
      bar.style.display = 'flex';
    } else {
      bar.style.display = 'none';
      if (selectAll) selectAll.checked = false;
    }
  }

  function clearBulkSelection() {
    document.querySelectorAll('.doc-checkbox').forEach(cb => cb.checked = false);
    const selectAll = document.getElementById('selectAllDocs');
    if (selectAll) selectAll.checked = false;
    updateBulkToolbar();
  }

  async function handleBulkAction(action) {
    const checkboxes = document.querySelectorAll('.doc-checkbox:checked');
    const filenames = Array.from(checkboxes).map(cb => cb.value);

    if (!filenames.length) {
      showToast('Please select at least one document.', 'warning');
      return;
    }

    const actionVerb = action === 'delete' ? 'permanently delete' : 'archive';
    const confirmMsg = `Are you sure you want to ${actionVerb} ${filenames.length} selected document(s)?\nThis will automatically update the AI vector database.`;

    if (!confirm(confirmMsg)) return;

    showLoadingOverlay(
      `Processing Bulk ${action === 'delete' ? 'Deletion' : 'Archiving'}...`,
      `Applying changes to ${filenames.length} files and re-indexing the AI vector store.`,
      false
    );

    try {
      const response = await fetch('/admin/bulk-documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: action, filenames: filenames })
      });

      const result = await response.json();
      if (response.ok && result.status === 'success') {
        showToast(result.message || 'Bulk operation successful!', 'success');
        setTimeout(() => {
          window.location.reload();
        }, 700);
      } else {
        hideLoadingOverlay();
        showToast(result.message || 'Bulk action failed.', 'error');
      }
    } catch (err) {
      hideLoadingOverlay();
      showToast('Network error executing bulk action.', 'error');
    }
  }

  // ==========================================================================
  // DOM EVENT BINDING & ZERO INLINE LISTENERS
  // ==========================================================================
  document.addEventListener('DOMContentLoaded', function () {
    // 1. Dropzone Drag & Drop & Staged Gallery
    const dropzone = document.getElementById('dropzoneBox');
    const fileInput = document.getElementById('fileInput');
    const addMoreBtn = document.getElementById('addMoreFilesBtn');
    const clearAllBtn = document.getElementById('clearAllFilesBtn');

    if (addMoreBtn && fileInput) {
      addMoreBtn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
      });
    }

    if (clearAllBtn) {
      clearAllBtn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        clearAllStagedFiles();
      });
    }

    if (dropzone && fileInput) {
      ['dragenter', 'dragover'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropzone.classList.add('dragover');
        }, false);
      });

      ['dragleave', 'drop'].forEach(name => {
        dropzone.addEventListener(name, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropzone.classList.remove('dragover');
        }, false);
      });

      dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleIncomingFiles(e.dataTransfer.files);
        }
      }, false);

      fileInput.addEventListener('click', function () {
        // Clear value on click before file picker opens so selecting the same file triggers 'change'
        this.value = '';
      });

      fileInput.addEventListener('change', function () {
        if (this.files && this.files.length > 0) {
          handleIncomingFiles(this.files);
        }
      });
    }

    // 2. Form Submissions
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadForm) {
      uploadForm.addEventListener('submit', handleUploadSubmit);
    }
    if (uploadBtn) {
      uploadBtn.addEventListener('click', function (e) {
        if (stagedFiles && stagedFiles.length > 0) {
          handleUploadSubmit(e);
        }
      });
    }

    const directNoticeForm = document.getElementById('directNoticeForm');
    if (directNoticeForm) {
      directNoticeForm.addEventListener('submit', handleDirectNoticeSubmit);
    }

    const reindexForm = document.getElementById('reindexForm');
    if (reindexForm) {
      reindexForm.addEventListener('submit', handleReindexSubmit);
    }

    const clearLogsForm = document.getElementById('clearLogsForm');
    if (clearLogsForm) {
      clearLogsForm.addEventListener('submit', function (e) {
        if (!confirm('Are you sure you want to clear all chat analytics and logs?')) {
          e.preventDefault();
        }
      });
    }

    const resolveGapForm = document.getElementById('resolveGapForm');
    if (resolveGapForm) {
      resolveGapForm.addEventListener('submit', function () {
        showLoadingOverlay('Adding Knowledge & Indexing...', 'Structuring content and updating vector search memory...');
      });
    }

    const editMetaForm = document.getElementById('editMetaForm');
    if (editMetaForm) {
      editMetaForm.addEventListener('submit', function () {
        showLoadingOverlay('Updating Metadata...', 'Synchronizing document lifecycle and vector search...');
      });
    }

    // 3. Select Dropdowns (doc_type & issuing_authority)
    document.querySelectorAll('select[name="doc_type"]').forEach(sel => {
      sel.addEventListener('change', function () {
        toggleCustomMetaType(this);
      });
    });

    document.querySelectorAll('select[name="issuing_authority"]').forEach(sel => {
      sel.addEventListener('change', function () {
        toggleCustomMetaAuth(this);
      });
    });

    // 4. Bulk Action Buttons
    const bulkArchiveBtn = document.getElementById('bulkArchiveBtn');
    if (bulkArchiveBtn) {
      bulkArchiveBtn.addEventListener('click', () => handleBulkAction('archive'));
    }

    const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
    if (bulkDeleteBtn) {
      bulkDeleteBtn.addEventListener('click', () => handleBulkAction('delete'));
    }

    const bulkClearBtn = document.getElementById('bulkClearBtn');
    if (bulkClearBtn) {
      bulkClearBtn.addEventListener('click', clearBulkSelection);
    }

    // 5. Modal Close Buttons
    const closeMetaModalBtn = document.getElementById('closeMetaModalBtn');
    if (closeMetaModalBtn) closeMetaModalBtn.addEventListener('click', closeMetaModal);

    const cancelMetaModalBtn = document.getElementById('cancelMetaModalBtn');
    if (cancelMetaModalBtn) cancelMetaModalBtn.addEventListener('click', closeMetaModal);

    const closeResolveModalBtn = document.getElementById('closeResolveModalBtn');
    if (closeResolveModalBtn) closeResolveModalBtn.addEventListener('click', closeResolveModal);

    const cancelResolveModalBtn = document.getElementById('cancelResolveModalBtn');
    if (cancelResolveModalBtn) cancelResolveModalBtn.addEventListener('click', closeResolveModal);

    // Modal Background Clicks & Escape Key
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        closeResolveModal();
        closeMetaModal();
      }
    });

    document.addEventListener('click', function(e) {
      const gapModal = document.getElementById('gapModal');
      if (gapModal && e.target === gapModal) closeResolveModal();
      const metaModal = document.getElementById('metaModal');
      if (metaModal && e.target === metaModal) closeMetaModal();
    });

    // 6. Global Delegated Clicks: Metadata edit, Resolve Gap, Delete forms, Alert dismissal
    document.addEventListener('click', function (e) {
      // Dismiss Alert Button
      const dismissBtn = e.target.closest('.alert-dismiss-btn') || e.target.closest('[data-dismiss="alert"]');
      if (dismissBtn) {
        const box = dismissBtn.closest('.alert-box, .analytics-alert, #routingSummarySection, .dismissible-alert');
        if (box) box.remove();
        return;
      }

      // Metadata Edit Button
      const metaBtn = e.target.closest('.meta-edit-btn');
      if (metaBtn) {
        openMetaModal(
          metaBtn.dataset.name,
          metaBtn.dataset.title,
          metaBtn.dataset.doctype,
          metaBtn.dataset.version,
          metaBtn.dataset.year,
          metaBtn.dataset.semester,
          metaBtn.dataset.authority,
          metaBtn.dataset.status,
          metaBtn.dataset.effective,
          metaBtn.dataset.expiry
        );
        return;
      }

      // Open Gap Modal Button
      const gapBtn = e.target.closest('.btn-resolve-gap');
      if (gapBtn) {
        openResolveModal(gapBtn.dataset.question || '', gapBtn.dataset.gapId || '');
        return;
      }

      // Quick Ajax Resolve Button
      const ajaxBtn = e.target.closest('.btn-resolve-ajax');
      if (ajaxBtn) {
        markGapResolvedAjax(ajaxBtn, ajaxBtn.dataset.gapId || '');
        return;
      }

      // Topic Pill Click
      const topicPill = e.target.closest('.topic-pill');
      if (topicPill) {
        openResolveModal(topicPill.dataset.question || '', '');
        return;
      }
    });

    // 7. Delete Document Form Submission Delegation
    document.addEventListener('submit', function (e) {
      const delForm = e.target.closest('.delete-doc-form');
      if (delForm) {
        const docName = delForm.dataset.docname || 'document';
        if (!confirmDelete(docName)) {
          e.preventDefault();
        }
      }
    });

    // 8. Initialize Live Search & Pagination
    window.docPaginator = new TablePaginator({
      tableId: 'kbDocsTable',
      searchInputId: 'docSearchInput',
      clearBtnId: 'clearDocSearch',
      rowsSelectId: 'docRowsPerPage',
      paginationNavId: 'docPaginationNav',
      paginationInfoId: 'docPaginationInfo',
      defaultRows: 25
    });

    window.logsPaginator = new TablePaginator({
      tableId: 'recentLogsTable',
      searchInputId: 'logsSearchInput',
      clearBtnId: 'clearLogsSearch',
      rowsSelectId: 'logsRowsPerPage',
      paginationNavId: 'logsPaginationNav',
      paginationInfoId: 'logsPaginationInfo',
      defaultRows: 15
    });

    window.gapsPaginator = new TablePaginator({
      tableId: 'knowledgeGapsTable',
      searchInputId: 'gapsSearchInput',
      clearBtnId: 'clearGapsSearch',
      rowsSelectId: 'gapsRowsPerPage',
      paginationNavId: 'gapsPaginationNav',
      paginationInfoId: 'gapsPaginationInfo',
      defaultRows: 10
    });

    const gapsFilterTabs = document.getElementById('gapsFilterTabs');
    if (gapsFilterTabs && window.gapsPaginator) {
      gapsFilterTabs.addEventListener('click', (e) => {
        const tab = e.target.closest('.btn-filter-tab');
        if (!tab) return;
        gapsFilterTabs.querySelectorAll('.btn-filter-tab').forEach(b => b.classList.remove('active'));
        tab.classList.add('active');
        window.gapsPaginator.currentFilter = tab.dataset.filter || 'all';
        window.gapsPaginator.filter();
      });
    }

    // 9. Bulk Checkbox Handlers
    const selectAll = document.getElementById('selectAllDocs');
    if (selectAll) {
      selectAll.addEventListener('change', function () {
        const isChecked = this.checked;
        const visibleRows = document.querySelectorAll('#kbDocsTable tbody tr');
        visibleRows.forEach(row => {
          if (row.style.display !== 'none') {
            const cb = row.querySelector('.doc-checkbox');
            if (cb) cb.checked = isChecked;
          }
        });
        updateBulkToolbar();
      });
    }

    const kbTable = document.getElementById('kbDocsTable');
    if (kbTable) {
      kbTable.addEventListener('change', function (e) {
        if (e.target.classList.contains('doc-checkbox')) {
          updateBulkToolbar();
        }
      });
    }
  });

  // Expose necessary global functions
  window.openMetaModal = openMetaModal;
  window.closeMetaModal = closeMetaModal;
  window.openResolveModal = openResolveModal;
  window.closeResolveModal = closeResolveModal;
  window.showToast = showToast;
  window.showLoadingOverlay = showLoadingOverlay;
  window.hideLoadingOverlay = hideLoadingOverlay;
})();
