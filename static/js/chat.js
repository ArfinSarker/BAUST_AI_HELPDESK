/**
 * ============================================================================
 * BAUST AI Help Desk - Student Chat Engine (chat.js)
 * 
 * Features:
 * 1. Real-Time Server-Sent Events (SSE) Streaming (/api/chat/stream)
 * 2. Strict XSS Protection via DOMPurify.sanitize(marked.parse(...))
 * 3. KaTeX Mathematical Formulas ($...$ and $$...$$) Auto-Rendering
 * 4. Highlight.js Code Syntax Highlighting with One-Click Copy Buttons
 * 5. Ultra-Smooth Decoupled Auto-Scroll Engine with User Scroll-Lock Detection
 * 6. Non-blocking Async Feedback (👍 / 👎) & Copy Text Handlers via Event Delegation
 * 7. Mobile Viewport Height Normalizer (--vh) & Double-Click Prevention
 * ============================================================================
 */

(function () {
  'use strict';

  // DOM Elements
  let chatMessages = null;
  let questionInput = null;
  let sendBtn = null;
  let stopBtn = null;
  let scrollBottomBtn = null;
  let typingIndicator = null;
  let toast = null;
  let chatForm = null;

  // Runtime State
  let conversationHistory = [];
  let isGenerating = false;
  let currentAbortController = null;
  let autoScroll = true;
  let previousScrollTop = 0;

  // ==========================================================================
  // TOAST NOTIFICATIONS
  // ==========================================================================
  function showToast(msg) {
    if (!toast) toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2200);
  }

  // ==========================================================================
  // ULTRA-SMOOTH AUTO-SCROLL DECOUPLING
  // ==========================================================================
  function isNearBottom(threshold = 80) {
    if (!chatMessages) return true;
    return (chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight) <= threshold;
  }

  function updateScrollState() {
    if (!chatMessages) return;
    const currentTop = chatMessages.scrollTop;
    const atBottom = isNearBottom(80);

    if (atBottom) {
      autoScroll = true;
      if (scrollBottomBtn) scrollBottomBtn.style.display = 'none';
    } else {
      // User is scrolling up or reading history
      if (currentTop < previousScrollTop || !isNearBottom(120)) {
        autoScroll = false;
      }
      if (scrollBottomBtn && !isNearBottom(140)) {
        scrollBottomBtn.style.display = 'flex';
      }
    }
    previousScrollTop = currentTop;
  }

  function scrollToBottom(force = false) {
    if (!chatMessages) return;
    if (force) {
      autoScroll = true;
      if (scrollBottomBtn) scrollBottomBtn.style.display = 'none';
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return;
    }
    if (autoScroll) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  function scrollDownSmooth() {
    if (!chatMessages) return;
    autoScroll = true;
    if (scrollBottomBtn) scrollBottomBtn.style.display = 'none';
    chatMessages.scrollTo({
      top: chatMessages.scrollHeight,
      behavior: 'smooth'
    });
  }

  // ==========================================================================
  // XSS-SAFE RICH CONTENT RENDERER (DOMPurify + Marked + KaTeX + Highlight.js)
  // ==========================================================================
  function sanitizeLinks(md) {
    if (!md) return md;
    // 1. Remove backticks around markdown links: `[text](url)` -> [text](url)
    let res = md.replace(/`(\[[^\]]+\]\([^\)]+\))`?/g, '$1');

    // 2. Fix multiple emails inside a single mailto: [a@b.com, c@d.com](mailto:...)
    res = res.replace(/\[([^\]]*@[^\]]*)\]\(mailto:([^)]*)\)/gi, function (match, label, href) {
      const combined = label + ' ' + href;
      const emailMatches = combined.match(/[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/g);
      if (emailMatches && emailMatches.length > 0) {
        const unique = Array.from(new Set(emailMatches.map(e => e.toLowerCase())));
        return unique.map(e => `[${e}](mailto:${e})`).join(', ');
      }
      return match;
    });

    // 3. Fix tel links with notes or multiple phones: [01759810833 (WhatsApp)](tel:...)
    res = res.replace(/\[([^\]]+)\]\(tel:([^)]*(?:\([^)]*\)[^)]*)*)\)/gi, function (match, label) {
      const parts = label.split(/[,/]/);
      const formatted = parts.map(p => {
        const trimmed = p.trim();
        const numMatch = trimmed.match(/(\+?\d[\d\s\-]{8,15}\d|\d{10,14})/);
        if (numMatch) {
          const rawNum = numMatch[1].trim();
          const cleanDigits = rawNum.replace(/[\s\-]/g, '');
          const note = trimmed.replace(rawNum, '').trim();
          return note ? `[${rawNum}](tel:${cleanDigits}) ${note}` : `[${rawNum}](tel:${cleanDigits})`;
        }
        return trimmed;
      });
      return formatted.join(', ');
    });

    // 4. Clean unnatural repetitive Bengali qualifications
    res = res.replace(/সি[\.\s]+এসই/g, 'সিএসই');
    res = res.replace(/ই[\.\s]+ই[\.\s]+ই/g, 'ইইই');
    res = res.replace(/আই[\.\s]+পি[\.\s]+ই/g, 'আইপিই');
    res = res.replace(/\bকেইউ\b/g, 'খুলনা বিশ্ববিদ্যালয় (KU)');
    res = res.replace(/\bজেইউ\b/g, 'জাহাঙ্গীরনগর বিশ্ববিদ্যালয় (JU)');
    res = res.replace(/\bডিইউ\b/g, 'ঢাকা বিশ্ববিদ্যালয় (DU)');
    res = res.replace(/\bসিইউ\b/g, 'চট্টগ্রাম বিশ্ববিদ্যালয় (CU)');
    res = res.replace(/\bআরইউ\b/g, 'রাজশাহী বিশ্ববিদ্যালয় (RU)');
    res = res.replace(/বি[\.\s]*এসসি\s*\(\s*(?:বি[\.\s]*এসসি[\.\s]*)?ইন\s*([^\)]+)\)/gi, 'বিএসসি ইন $1');
    res = res.replace(/এম[\.\s]*এসসি\s*\(\s*(?:এম[\.\s]*এসসি[\.\s]*)?ইন\s*([^\)]+)\)/gi, 'এমএসসি ইন $1');
    res = res.replace(/(বি[\.\s]*এসসি|এম[\.\s]*এসসি)\s*\(\s*\1\s*[,;]?\s*([^\)]*)\)/gi, '$1 ($2)');

    return res;
  }

  function renderRichContent(container, markdownText, isComplete = false) {
    if (!container) return;
    if (!markdownText) {
      container.innerHTML = '';
      return;
    }

    // Clean broken link syntax and separate multiple contact links
    const cleanMd = sanitizeLinks(markdownText);

    // 1. Convert Markdown to HTML
    let rawHtml = (window.marked && typeof window.marked.parse === 'function')
      ? window.marked.parse(cleanMd)
      : cleanMd;

    // 2. Strict XSS Sanitization via DOMPurify
    let cleanHtml = (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function')
      ? window.DOMPurify.sanitize(rawHtml)
      : rawHtml;

    container.innerHTML = cleanHtml;

    // 3. Highlight.js Code Highlighting & Copy Button (Fired on complete or final chunks)
    if (window.hljs && isComplete) {
      container.querySelectorAll('pre code').forEach((block) => {
        try {
          window.hljs.highlightElement(block);
        } catch (e) {}

        const parent = block.parentElement;
        if (parent && !parent.querySelector('.code-copy-btn')) {
          const copyBtn = document.createElement('button');
          copyBtn.className = 'code-copy-btn';
          copyBtn.type = 'button';
          copyBtn.innerHTML = '📋 Copy';
          copyBtn.addEventListener('click', function () {
            navigator.clipboard.writeText(block.innerText).then(() => {
              copyBtn.innerHTML = '✅ Copied!';
              setTimeout(() => { copyBtn.innerHTML = '📋 Copy'; }, 2000);
            });
          });
          parent.style.position = 'relative';
          parent.appendChild(copyBtn);
        }
      });
    }

    // 4. KaTeX Math Equations Rendering ($...$ and $$...$$)
    if (window.renderMathInElement && isComplete) {
      try {
        window.renderMathInElement(container, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
          ],
          throwOnError: false
        });
      } catch (mathErr) {
        console.debug('KaTeX parse notice:', mathErr);
      }
    }
  }

  // ==========================================================================
  // UI GENERATING & DISABLED STATES (PREVENTS DOUBLE-CLICKS)
  // ==========================================================================
  function setGeneratingUI() {
    isGenerating = true;
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.style.display = 'none';
    }
    if (stopBtn) {
      stopBtn.style.display = 'inline-flex';
    }
    if (typingIndicator) {
      typingIndicator.style.display = 'flex';
    }
    if (questionInput) {
      questionInput.placeholder = 'AI উত্তর তৈরি করছে... থামাতে ⏹ চাপুন';
    }
  }

  function resetInputUI() {
    isGenerating = false;
    if (stopBtn) {
      stopBtn.style.display = 'none';
    }
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.style.display = 'inline-flex';
    }
    if (typingIndicator) {
      typingIndicator.style.display = 'none';
    }
    if (questionInput) {
      questionInput.placeholder = 'ভর্তি, শিক্ষক, ফি বা যেকোনো কিছু জিজ্ঞেস করুন...';
      questionInput.disabled = false;
      questionInput.focus();
    }
  }

  function stopGenerating() {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
    }
    isGenerating = false;
    resetInputUI();
    showToast('⏹️ উত্তর জেনারেশন থামানো হয়েছে');
  }

  // ==========================================================================
  // MESSAGE RENDERING
  // ==========================================================================
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="avatar user">You</div>
      <div class="bubble user">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom(true);
  }

  function copyText(button) {
    const bubble = button.closest('.bubble');
    if (!bubble) return;
    const clone = bubble.cloneNode(true);
    const actions = clone.querySelector('.bubble-actions');
    if (actions) actions.remove();

    const textToCopy = clone.innerText.trim();
    navigator.clipboard.writeText(textToCopy).then(() => {
      showToast('✅ উত্তর কপি করা হয়েছে!');
    });
  }

  async function sendFeedback(button, type) {
    const bubble = button.closest('.bubble');
    const actions = button.closest('.bubble-actions');
    const row = button.closest('.message-row.ai');

    // Find original question
    let question = '';
    if (row && row.dataset.question) {
      question = row.dataset.question;
    } else if (row) {
      let prev = row.previousElementSibling;
      while (prev) {
        if (prev.classList.contains('user')) {
          const userBubble = prev.querySelector('.bubble.user');
          if (userBubble) question = userBubble.innerText.trim();
          break;
        }
        prev = prev.previousElementSibling;
      }
    }

    // Find answer text without buttons
    let answerText = '';
    const streamDiv = bubble ? bubble.querySelector('.stream-content') : null;
    if (streamDiv && streamDiv.innerText.trim()) {
      answerText = streamDiv.innerText.trim();
    } else if (bubble) {
      const clone = bubble.cloneNode(true);
      const cloneActions = clone.querySelector('.bubble-actions');
      if (cloneActions) cloneActions.remove();
      answerText = clone.innerText.trim();
    }

    const likeBtn = actions ? actions.querySelector('.like-btn') : null;
    const dislikeBtn = actions ? actions.querySelector('.dislike-btn') : null;
    const logId = (row && row.dataset.logId) ? parseInt(row.dataset.logId, 10) : null;

    // Optimistic UI update
    if (type === 'like') {
      if (likeBtn) likeBtn.classList.add('active-good');
      if (dislikeBtn) dislikeBtn.classList.remove('active-bad');
      showToast('👍 মতামতের জন্য ধন্যবাদ! (Liked)');
    } else {
      if (dislikeBtn) dislikeBtn.classList.add('active-bad');
      if (likeBtn) likeBtn.classList.remove('active-good');
      showToast('👎 আপনার ফিডব্যাক রেকর্ড করা হয়েছে।');
    }

    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          log_id: logId,
          question: question || 'Student Query',
          answer: answerText,
          feedback: type
        })
      });
    } catch (err) {
      console.error('Feedback submission error:', err);
    }
  }

  function clearChat() {
    if (confirm('বর্তমান কথোপকথন মুছে ফেলতে চান?')) {
      if (isGenerating) stopGenerating();
      conversationHistory = [];
      const welcome = document.getElementById('welcomeHero');
      chatMessages.innerHTML = '';
      if (welcome) {
        chatMessages.appendChild(welcome);
      }
      if (window.speechSynthesis) window.speechSynthesis.cancel();
      scrollToBottom(true);
    }
  }

  // ==========================================================================
  // REAL-TIME SSE STREAMING CHAT ENGINE
  // ==========================================================================
  async function sendMessage(question) {
    if (!question || !question.trim() || isGenerating) return;

    const trimmedQuestion = question.trim();
    if (questionInput) questionInput.value = '';

    // Reset scroll lock for new message
    autoScroll = true;

    // Render user question
    appendUserMessage(trimmedQuestion);
    conversationHistory.push({ role: 'user', content: trimmedQuestion });

    // Create empty AI message container for live token stream
    const row = document.createElement('div');
    row.className = 'message-row ai';
    row.dataset.question = trimmedQuestion;
    row.innerHTML = `
      <div class="avatar ai">AI</div>
      <div class="bubble ai">
        <div class="stream-content"></div>
        <div class="bubble-actions" style="display: none;">
          <button type="button" class="action-icon-btn feedback-btn like-btn" title="পছন্দ হয়েছে (Helpful)">👍 Like</button>
          <button type="button" class="action-icon-btn feedback-btn dislike-btn" title="অপছন্দ হয়েছে (Not helpful)">👎 Dislike</button>
          <button type="button" class="action-icon-btn copy-btn" title="উত্তর কপি করুন">📋 Copy</button>
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
    scrollToBottom(true);

    const streamContentDiv = row.querySelector('.stream-content');
    const bubbleActions = row.querySelector('.bubble-actions');

    setGeneratingUI();
    scrollToBottom(true);

    let accumulatedText = '';
    currentAbortController = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: currentAbortController.signal,
        body: JSON.stringify({
          question: trimmedQuestion,
          history: conversationHistory
        })
      });

      if (typingIndicator) typingIndicator.style.display = 'none';

      if (!response.ok || !response.body) {
        throw new Error('Stream response error: ' + response.statusText);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data: ')) {
            const jsonStr = trimmedLine.slice(6);
            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.type === 'token' && parsed.content) {
                accumulatedText += parsed.content;
                renderRichContent(streamContentDiv, accumulatedText, false);
                scrollToBottom(false);
              } else if (parsed.type === 'done') {
                if (parsed.raw_answer) {
                  accumulatedText = sanitizeLinks(parsed.raw_answer);
                }
                if (parsed.log_id) {
                  row.dataset.logId = parsed.log_id;
                }
              } else if (parsed.type === 'error') {
                accumulatedText = `⚠️ ${parsed.message || 'Error'}`;
                const safeErr = (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function')
                  ? window.DOMPurify.sanitize(accumulatedText)
                  : accumulatedText;
                streamContentDiv.innerHTML = `<p style="color: #EF4444;">${safeErr}</p>`;
              }
            } catch (e) {}
          }
        }
      }

      // Finalize formatting
      if (accumulatedText) {
        renderRichContent(streamContentDiv, accumulatedText, true);
      }

      // Setup Actions with smooth fade-in
      if (bubbleActions) {
        bubbleActions.style.display = 'flex';
        requestAnimationFrame(() => bubbleActions.classList.add('visible'));
      }
      scrollToBottom(false);
      conversationHistory.push({ role: 'assistant', content: accumulatedText });

    } catch (err) {
      if (err.name === 'AbortError') {
        // User clicked stop
        if (accumulatedText) {
          renderRichContent(streamContentDiv, accumulatedText, true);
          if (bubbleActions) bubbleActions.style.display = 'flex';
          conversationHistory.push({ role: 'assistant', content: accumulatedText });
        }
      } else {
        console.error('Chat error:', err);
        if (typingIndicator) typingIndicator.style.display = 'none';
        streamContentDiv.innerHTML = `<p style="color: #EF4444;">⚠️ সার্ভারের সাথে সংযোগ স্থাপন করা সম্ভব হয়নি। অনুগ্রহ করে আবার চেষ্টা করুন।</p>`;
      }
    } finally {
      resetInputUI();
      currentAbortController = null;
      scrollToBottom(false);
    }
  }

  // ==========================================================================
  // EVENT BINDINGS (ZERO INLINE EVENT LISTENERS)
  // ==========================================================================
  document.addEventListener('DOMContentLoaded', function () {
    chatMessages = document.getElementById('chatMessages');
    questionInput = document.getElementById('questionInput');
    sendBtn = document.getElementById('sendBtn');
    stopBtn = document.getElementById('stopBtn');
    scrollBottomBtn = document.getElementById('scrollBottomBtn');
    typingIndicator = document.getElementById('typingIndicator');
    toast = document.getElementById('toast');
    chatForm = document.getElementById('chatForm');

    // 1. Chat Form Submit (Prevents Double-Clicks)
    if (chatForm) {
      chatForm.addEventListener('submit', function (event) {
        event.preventDefault();
        if (isGenerating) {
          stopGenerating();
          return;
        }
        if (questionInput) {
          sendMessage(questionInput.value);
        }
      });
    }

    // 2. Stop Button Listener
    if (stopBtn) {
      stopBtn.addEventListener('click', stopGenerating);
    }

    // 3. Clear Chat Button Listener
    const clearBtn = document.getElementById('clearChatBtn') || document.querySelector('.clear-chat-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', clearChat);
    }

    // 4. Scroll to Bottom Button Listener
    if (scrollBottomBtn) {
      scrollBottomBtn.addEventListener('click', scrollDownSmooth);
    }

    // 5. Scroll State Listeners
    if (chatMessages) {
      chatMessages.addEventListener('scroll', updateScrollState, { passive: true });

      chatMessages.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
          autoScroll = false;
          if (scrollBottomBtn) scrollBottomBtn.style.display = 'flex';
        }
      }, { passive: true });

      chatMessages.addEventListener('touchmove', () => {
        if (!isNearBottom(80)) {
          autoScroll = false;
          if (scrollBottomBtn) scrollBottomBtn.style.display = 'flex';
        }
      }, { passive: true });

      // 6. Event Delegation for dynamically inserted buttons (Like, Dislike, Copy)
      chatMessages.addEventListener('click', function (e) {
        const copyBtn = e.target.closest('.copy-btn');
        if (copyBtn) {
          copyText(copyBtn);
          return;
        }
        const likeBtn = e.target.closest('.like-btn');
        if (likeBtn) {
          sendFeedback(likeBtn, 'like');
          return;
        }
        const dislikeBtn = e.target.closest('.dislike-btn');
        if (dislikeBtn) {
          sendFeedback(dislikeBtn, 'dislike');
          return;
        }
      });
    }

    // 7. Input Focus & Escape Key Handlers
    if (questionInput) {
      questionInput.addEventListener('focus', () => {
        setTimeout(() => scrollToBottom(true), 300);
      });
    }

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isGenerating) {
        stopGenerating();
      }
    });

    // 8. Mobile Viewport Height Normalizer
    function setAppHeight() {
      const doc = document.documentElement;
      doc.style.setProperty('--app-height', `${window.innerHeight}px`);
    }
    window.addEventListener('resize', setAppHeight);
    setAppHeight();

    // Initial scroll & focus
    scrollToBottom(true);
    if (window.innerWidth > 640 && questionInput) {
      questionInput.focus();
    }
  });

  // Expose helpers if needed externally
  window.sendMessage = sendMessage;
  window.stopGenerating = stopGenerating;
  window.clearChat = clearChat;
  window.scrollDownSmooth = scrollDownSmooth;
})();

