(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    app: document.querySelector(".app"),
    messages: $("messages"), empty: $("empty"), input: $("input"), send: $("sendBtn"),
    attachments: $("attachments"), menu: $("menu"), banner: $("banner"),
    camModal: $("cameraModal"), camVideo: $("camVideo"), camCanvas: $("camCanvas"), camError: $("camError"),
  };

  const MAX_FILES = 6;
  const MAX_IMAGES = 4;

  const state = {
    pending: [],           // attachments for the next message: {file, kind: "pdf"|"image", url?}
    docIds: new Set(),     // documents indexed for THIS chat (PDFs and saved photos)
    history: [],           // [{role, content}]
    busy: false,
    stream: null,
    facing: "environment",
  };

  // ------------------------------------------------------------ utils
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtSize = (b) => (b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
  const isPdf = (f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name);
  const isImage = (f) => f.type.startsWith("image/");

  function toast(text, isErr = false, ms = 4500) {
    const t = document.createElement("div");
    t.className = "toast" + (isErr ? " err" : "");
    t.textContent = text;
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  function openLightbox(src) {
    $("lightboxImg").src = src;
    $("lightbox").hidden = false;
  }
  $("lightbox").onclick = () => ($("lightbox").hidden = true);

  // ------------------------------------------------------------ tiny markdown renderer (XSS-safe)
  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/\[(\d{1,3})\]/g, '<button class="cite" data-n="$1" title="Show source">$1</button>');
  }

  function renderMarkdown(src) {
    const codes = [];
    src = src.replace(/```(\w*)\n?([\s\S]*?)(?:```|$)/g, (_, _l, code) => {
      codes.push(code.replace(/\n$/, ""));
      return `\u0000${codes.length - 1}\u0000`;
    });
    const lines = esc(src).split("\n");
    const out = [];
    let i = 0;
    const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l || "");
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

    while (i < lines.length) {
      const line = lines[i];
      let m;
      if (!line.trim()) { i++; continue; }
      if ((m = line.match(/^\u0000(\d+)\u0000$/))) {
        out.push(`<pre><code>${esc(codes[+m[1]])}</code></pre>`); i++; continue;
      }
      if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
        out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); i++; continue;
      }
      if (/^\s*([-*_]){3,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
      if (line.includes("|") && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
        out.push(
          "<table><thead><tr>" + head.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
          rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") + "</tbody></table>"
        );
        continue;
      }
      if (/^\s*([-*•])\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]\s+/.test(line);
        const re = ordered ? /^\s*\d+[.)]\s+(.*)$/ : /^\s*[-*•]\s+(.*)$/;
        const items = [];
        while (i < lines.length && (m = lines[i].match(re))) { items.push(`<li>${inline(m[1])}</li>`); i++; }
        out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
        continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*•]\s|\s*\d+[.)]\s|\u0000\d+\u0000$)/.test(lines[i])) para.push(lines[i++]);
      if (!para.length) { para.push(lines[i++]); }
      out.push(`<p>${inline(para.join("<br>"))}</p>`);
    }
    return out.join("");
  }

  // ------------------------------------------------------------ attachments (PDF, gallery, camera)
  function addAttachments(files) {
    for (const file of files) {
      const pdf = isPdf(file);
      const img = !pdf && isImage(file);
      if (!pdf && !img) { toast(`${file.name}: only PDFs and images are supported.`, true); continue; }
      if (state.pending.length >= MAX_FILES) { toast(`You can attach up to ${MAX_FILES} files per message.`); break; }
      if (img && state.pending.filter((a) => a.kind === "image").length >= MAX_IMAGES) {
        toast(`You can attach up to ${MAX_IMAGES} photos per message.`);
        continue;
      }
      state.pending.push({ file, kind: pdf ? "pdf" : "image", url: img ? URL.createObjectURL(file) : null });
    }
    renderAttachments();
  }

  function fileChipHtml(name, size, extra = "") {
    return `<div class="file-icon">PDF</div><div class="file-info"><div class="file-name" title="${esc(name)}">${esc(name)}</div><div class="file-meta">PDF &middot; ${fmtSize(size)}</div></div>${extra}`;
  }

  function renderAttachments() {
    els.attachments.innerHTML = state.pending
      .map((a, i) =>
        a.kind === "image"
          ? `<div class="attach"><img src="${a.url}" alt="attached photo" data-zoom="${i}"><button class="rm" data-rm="${i}" aria-label="Remove">&times;</button></div>`
          : `<div class="file-chip">${fileChipHtml(a.file.name, a.file.size, `<button class="rm" data-rm="${i}" aria-label="Remove">&times;</button>`)}</div>`
      )
      .join("");
    updateSend();
  }

  els.attachments.addEventListener("click", (e) => {
    const rm = e.target.closest("[data-rm]");
    const zoom = e.target.closest("[data-zoom]");
    if (rm) {
      const [a] = state.pending.splice(+rm.dataset.rm, 1);
      if (a.url) URL.revokeObjectURL(a.url);
      renderAttachments();
    } else if (zoom) openLightbox(state.pending[+zoom.dataset.zoom].url);
  });

  $("galleryInput").onchange = (e) => { addAttachments(e.target.files); e.target.value = ""; };
  $("captureInput").onchange = (e) => { addAttachments(e.target.files); e.target.value = ""; };
  $("pdfInput").onchange = (e) => { addAttachments(e.target.files); e.target.value = ""; };

  els.input.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])].filter((f) => isImage(f) || isPdf(f));
    if (files.length) { e.preventDefault(); addAttachments(files); }
  });

  // drag & drop anywhere on the page
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types?.includes("Files")) { dragDepth++; els.app.classList.add("drag"); } });
  window.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) els.app.classList.remove("drag"); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    els.app.classList.remove("drag");
    if (e.dataTransfer?.files?.length) addAttachments([...e.dataTransfer.files]);
  });

  // + menu
  $("plusBtn").onclick = (e) => { e.stopPropagation(); els.menu.hidden = !els.menu.hidden; };
  document.addEventListener("click", (e) => { if (!els.menu.contains(e.target)) els.menu.hidden = true; });
  $("menuGallery").onclick = () => { els.menu.hidden = true; $("galleryInput").click(); };
  $("menuPdf").onclick = () => { els.menu.hidden = true; $("pdfInput").click(); };
  $("menuCamera").onclick = () => { els.menu.hidden = true; openCamera(); };

  // ------------------------------------------------------------ camera
  async function startStream() {
    stopStream();
    els.camError.hidden = true;
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: state.facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      els.camVideo.srcObject = state.stream;
    } catch (err) {
      els.camError.hidden = false;
      els.camError.textContent =
        err.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access in your browser settings, or choose from gallery instead."
        : err.name === "NotFoundError" ? "No camera was found on this device."
        : "Could not start the camera: " + err.message;
    }
  }
  function stopStream() {
    state.stream?.getTracks().forEach((t) => t.stop());
    state.stream = null;
    els.camVideo.srcObject = null;
  }
  function openCamera() {
    // Without getUserMedia (e.g. plain http on a phone) fall back to the native camera app.
    if (!navigator.mediaDevices?.getUserMedia) return $("captureInput").click();
    els.camModal.hidden = false;
    startStream();
  }
  function closeCamera() { stopStream(); els.camModal.hidden = true; }
  $("camClose").onclick = $("camCancel").onclick = closeCamera;
  $("camSwitch").onclick = () => { state.facing = state.facing === "environment" ? "user" : "environment"; startStream(); };
  $("camShoot").onclick = () => {
    const v = els.camVideo;
    if (!v.videoWidth) return;
    els.camCanvas.width = v.videoWidth;
    els.camCanvas.height = v.videoHeight;
    els.camCanvas.getContext("2d").drawImage(v, 0, 0);
    els.camCanvas.toBlob((blob) => {
      if (!blob) return;
      addAttachments([new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" })]);
      closeCamera();
      els.input.focus();
    }, "image/jpeg", 0.92);
  };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeCamera(); $("lightbox").hidden = true; } });

  // ------------------------------------------------------------ chat
  function updateSend() {
    const hasContent = els.input.value.trim() || state.pending.length;
    els.send.disabled = state.busy || !hasContent;
  }
  els.input.addEventListener("input", () => {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 180) + "px";
    updateSend();
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!els.send.disabled) sendMessage(); }
  });
  els.send.onclick = () => sendMessage();
  document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => { els.input.value = c.dataset.q; els.input.dispatchEvent(new Event("input")); els.input.focus(); }));

  function scrollDown(force = false) {
    const m = els.messages;
    if (force || m.scrollHeight - m.scrollTop - m.clientHeight < 160) m.scrollTop = m.scrollHeight;
  }

  function addMessage(role, html, attached = []) {
    els.empty?.remove();
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const items = attached
      .map((a) =>
        a.kind === "image"
          ? `<img src="${a.url}" alt="attached photo">`
          : `<a class="file-chip" data-fname="${esc(a.file.name)}" target="_blank" rel="noopener">${fileChipHtml(a.file.name, a.file.size)}</a>`
      )
      .join("");
    const box = items ? `<div class="msg-attachments">${items}</div>` : "";
    wrap.innerHTML = `${role === "assistant" ? '<div class="avatar">&#9673;</div>' : ""}<div class="bubble">${box}<div class="content">${html}</div></div>`;
    els.messages.appendChild(wrap);
    wrap.querySelectorAll(".msg-attachments img").forEach((im) => (im.onclick = () => openLightbox(im.src)));
    return wrap;
  }

  const spinner = (text) => `<div class="status"><div class="spinner"></div><span>${esc(text)}</span></div>`;

  async function uploadPdfs(pdfs, userMsg) {
    const fd = new FormData();
    pdfs.forEach((a) => fd.append("files", a.file, a.file.name));
    const res = await fetch("/api/documents", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    const errors = [];
    for (const r of await res.json()) {
      if (!r.ok) { errors.push(`${r.name}: ${r.error}`); continue; }
      state.docIds.add(r.document.id);
      const chip = [...userMsg.querySelectorAll("a.file-chip")].find((c) => c.dataset.fname === r.name);
      if (chip) chip.href = `/api/documents/${r.document.id}/file`;
    }
    return errors;
  }

  async function sendMessage() {
    const text = els.input.value.trim();
    if ((!text && !state.pending.length) || state.busy) return;

    const attached = state.pending.splice(0);
    const pdfs = attached.filter((a) => a.kind === "pdf");
    const images = attached.filter((a) => a.kind === "image");
    // Photos are meaningless without a question; PDFs alone default to a summary request.
    const question = text || (pdfs.length && !images.length ? "Summarize this document." : "What does this show?");
    renderAttachments();
    els.input.value = "";
    els.input.style.height = "auto";
    state.busy = true;
    updateSend();

    const userMsg = addMessage("user", text ? `<p>${esc(text).replace(/\n/g, "<br>")}</p>` : "", attached);
    const bot = addMessage("assistant", spinner(pdfs.length ? `Reading ${pdfs.length === 1 ? pdfs[0].file.name : pdfs.length + " PDFs"}...` : "Thinking..."));
    const content = bot.querySelector(".content");
    if (!text) userMsg.querySelector(".content").remove();
    scrollDown(true);

    let answer = "";
    let sources = [];
    let failed = false;

    const fail = (message) => {
      failed = true;
      content.innerHTML = `<div class="error-box">${esc(message)}</div>`;
    };

    // 1) index any attached PDFs so they become part of this chat
    if (pdfs.length) {
      try {
        const errors = await uploadPdfs(pdfs, userMsg);
        if (errors.length) fail("I couldn't read: " + errors.join(" | "));
      } catch (err) {
        fail(err.message);
      }
    }

    if (!failed) {
      content.innerHTML = spinner("Thinking...");
      const fd = new FormData();
      fd.append("message", question);
      fd.append("history", JSON.stringify(state.history));
      fd.append("doc_ids", JSON.stringify([...state.docIds]));
      fd.append("remember", "true"); // photos are indexed so follow-up questions can use them
      images.forEach((a) => fd.append("images", a.file, a.file.name));

      const handlers = {
        status: (d) => { if (!answer) content.innerHTML = spinner(d.text); },
        sources: (d) => { sources = d; },
        delta: (d) => {
          answer += d.text;
          content.innerHTML = renderMarkdown(answer) + '<span class="typing"></span>';
          scrollDown();
        },
        done: (d) => {
          d.saved?.forEach((s) => s.id && state.docIds.add(s.id));
          d.saved?.filter((s) => s.error).forEach((s) => toast(`Could not index photo: ${s.error}`, true));
          finish();
        },
        error: (d) => fail(d.message),
      };

      function finish() {
        content.innerHTML = renderMarkdown(answer || "_(no answer)_");
        const cited = new Set([...answer.matchAll(/\[(\d{1,3})\]/g)].map((m) => +m[1]));
        const used = sources.filter((s) => cited.has(s.n));
        if (used.length) {
          const box = document.createElement("div");
          box.className = "sources";
          box.innerHTML = `<div class="sources-title">Sources</div>` + used.map((s) => {
            const label = s.kind === "pdf" ? `p. ${s.page}` : "photo";
            return `<details class="source" data-n="${s.n}">
              <summary><span class="cite">${s.n}</span><span class="source-name">${esc(s.name)}</span><span class="source-page">${label}</span></summary>
              <div class="source-body">${esc(s.text)}${s.kind === "pdf" ? `\n\n<a class="source-open" href="/api/documents/${s.doc_id}/file#page=${s.page}" target="_blank">Open PDF at page ${s.page}</a>` : ""}</div>
            </details>`;
          }).join("");
          content.appendChild(box);
        }
        const actions = document.createElement("div");
        actions.className = "msg-actions";
        actions.innerHTML = '<button class="mini">Copy</button>';
        actions.firstChild.onclick = () => { navigator.clipboard.writeText(answer); toast("Copied"); };
        content.appendChild(actions);
      }

      try {
        const res = await fetch("/api/chat", { method: "POST", body: fd });
        if (!res.ok) {
          let msg = `Request failed (${res.status})`;
          try { const j = await res.json(); msg = j.error || j.detail || msg; } catch {}
          fail(msg);
        } else {
          const reader = res.body.getReader();
          const dec = new TextDecoder();
          let buf = "";
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf("\n\n")) >= 0) {
              const raw = buf.slice(0, idx);
              buf = buf.slice(idx + 2);
              const ev = /^event: (.*)$/m.exec(raw)?.[1];
              const data = /^data: (.*)$/m.exec(raw)?.[1];
              if (ev && data && handlers[ev]) handlers[ev](JSON.parse(data));
            }
          }
          if (!failed && !answer) finish();
        }
      } catch (err) {
        fail("Connection lost: " + err.message);
      }
    }

    if (!failed) {
      const names = attached.map((a) => a.file.name);
      const note = names.length ? `[Attached: ${names.join(", ")}]\n` : "";
      state.history.push({ role: "user", content: note + question }, { role: "assistant", content: answer });
      state.history = state.history.slice(-12);
    }
    state.busy = false;
    updateSend();
    scrollDown(true);
  }

  // citation click -> open + flash the source card inside that message
  els.messages.addEventListener("click", (e) => {
    const cite = e.target.closest("button.cite");
    if (!cite) return;
    const card = cite.closest(".msg")?.querySelector(`.source[data-n="${cite.dataset.n}"]`);
    if (!card) return;
    card.open = true;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
  });

  // ------------------------------------------------------------ chat lifecycle
  // Files belong to the chat that received them: starting a new chat (or closing the page) removes them.
  function discardChatDocuments(useBeacon = false) {
    if (!state.docIds.size) return;
    const body = JSON.stringify({ ids: [...state.docIds] });
    state.docIds.clear();
    if (useBeacon && navigator.sendBeacon) {
      navigator.sendBeacon("/api/documents/delete", new Blob([body], { type: "application/json" }));
    } else {
      fetch("/api/documents/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true }).catch(() => {});
    }
  }
  window.addEventListener("pagehide", () => discardChatDocuments(true));

  $("clearChat").onclick = () => {
    if (state.busy) return;
    discardChatDocuments();
    state.pending.forEach((a) => a.url && URL.revokeObjectURL(a.url));
    state.pending = [];
    state.history = [];
    renderAttachments();
    els.messages.innerHTML = "";
    els.messages.appendChild(els.empty);
  };

  // ------------------------------------------------------------ boot
  (async function init() {
    try {
      const h = await (await fetch("/api/health")).json();
      if (!h.api_key_configured) {
        els.banner.hidden = false;
        els.banner.textContent = "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, and restart the server.";
      }
    } catch {}
  })();
})();
