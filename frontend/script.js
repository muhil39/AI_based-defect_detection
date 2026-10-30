/**
 * AI-Based Surface Defect Detection - Frontend Controller
 * Manages webcam capture, sequential AI inference, voice alerts, and SQLite history.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const videoFeed = document.getElementById("webcam-feed");
  const capturedCanvas = document.getElementById("captured-canvas");
  const cameraPlaceholder = document.getElementById("camera-placeholder");
  const scanline = document.getElementById("scanline");
  const liveClock = document.getElementById("live-clock");

  // Telemetry indicators
  const aiStatusDot = document.getElementById("ai-status-dot");
  const aiStatusText = document.getElementById("ai-status-text");
  const camStatusDot = document.getElementById("cam-status-dot");
  const camStatusText = document.getElementById("cam-status-text");
  const systemBannerText = document.getElementById("system-banner-text");
  const hudResolution = document.getElementById("hud-resolution");

  // Buttons
  const btnStartCamera = document.getElementById("btn-start-camera");
  const btnStopCamera = document.getElementById("btn-stop-camera");
  const btnCapture = document.getElementById("btn-capture");
  const manualFileInput = document.getElementById("manual-file-input");
  const btnResumeStream = document.getElementById("btn-resume-stream");
  const btnTestSpeaker = document.getElementById("btn-test-speaker");
  const voiceAlertToggle = document.getElementById("voice-alert-toggle");

  // Navigation Tabs
  const tabInspectionBtn = document.getElementById("tab-inspection-btn");
  const tabHistoryBtn = document.getElementById("tab-history-btn");
  const viewInspection = document.getElementById("view-inspection");
  const viewHistory = document.getElementById("view-history");
  const historyCounter = document.getElementById("history-counter");

  // Material Display Elements
  const materialBadge = document.getElementById("material-badge");
  const materialIconPlate = document.getElementById("material-icon-plate");
  const materialIcon = document.getElementById("material-icon");
  const materialResultText = document.getElementById("material-result-text");
  const materialSubtext = document.getElementById("material-subtext");
  const materialConfidenceLabel = document.getElementById("material-confidence-label");
  const materialMeterFill = document.getElementById("material-meter-fill");

  // Crack Display Elements
  const inspectionResultBox = document.getElementById("inspection-result-box");
  const resultTag = document.getElementById("result-tag");
  const resultHeadline = document.getElementById("result-headline");
  const resultDescription = document.getElementById("result-description");
  const crackConfidenceLabel = document.getElementById("crack-confidence-label");
  const crackMeterFill = document.getElementById("crack-meter-fill");
  const lblInspectionId = document.getElementById("lbl-inspection-id");
  const lblInspectionTime = document.getElementById("lbl-inspection-time");
  const lblInspectionVerdict = document.getElementById("lbl-inspection-verdict");

  // Stepper Nodes
  const stepItems = [
    document.getElementById("step-1"),
    document.getElementById("step-2"),
    document.getElementById("step-3"),
    document.getElementById("step-4"),
    document.getElementById("step-5")
  ];

  // History & Stats
  const historyTableBody = document.getElementById("history-table-body");
  const statTotal = document.getElementById("stat-total");
  const statMetal = document.getElementById("stat-metal");
  const statCracks = document.getElementById("stat-cracks");
  const statRate = document.getElementById("stat-rate");
  const btnRefreshHistory = document.getElementById("btn-refresh-history");
  const btnClearHistory = document.getElementById("btn-clear-history");

  // Image Modal
  const imageModal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const modalTitle = document.getElementById("modal-title");
  const modalCaption = document.getElementById("modal-caption");
  const modalClose = document.getElementById("modal-close");

  // State
  let streamInstance = null;
  let isInspecting = false;
  let lastAnnouncedId = null;

  // 1. LIVE CLOCK UPDATE
  function updateClock() {
    const now = new Date();
    liveClock.textContent = now.toLocaleTimeString();
  }
  setInterval(updateClock, 1000);
  updateClock();

  // 2. TOAST NOTIFICATIONS
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // 3. STEPPER PROGRESS HELPER
  function setStepper(activeStep) {
    stepItems.forEach((item, index) => {
      const stepNum = index + 1;
      item.classList.remove("active", "completed");
      if (stepNum < activeStep) {
        item.classList.add("completed");
      } else if (stepNum === activeStep) {
        item.classList.add("active");
      }
    });
  }

  function resetStepper() {
    stepItems.forEach(item => item.classList.remove("active", "completed"));
  }

  // 4. WEBCAM CONTROLS
  async function startCamera() {
    try {
      showToast("Accessing webcam...", "info");
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamInstance = stream;
      videoFeed.srcObject = stream;
      videoFeed.style.display = "block";
      capturedCanvas.style.display = "none";
      cameraPlaceholder.style.display = "none";

      videoFeed.onloadedmetadata = () => {
        videoFeed.play();
        hudResolution.textContent = `RES: ${videoFeed.videoWidth || 640} x ${videoFeed.videoHeight || 480}`;
      };

      // Update UI Controls
      btnStartCamera.disabled = true;
      btnStopCamera.disabled = false;
      btnCapture.disabled = false;
      btnResumeStream.style.display = "none";

      camStatusDot.className = "status-dot online";
      camStatusText.textContent = "STREAMING (LIVE)";
      systemBannerText.textContent = "Webcam streaming active. Position a component and click Capture & Inspect.";
      showToast("Camera stream connected successfully.", "success");
    } catch (err) {
      console.error("Camera error:", err);
      camStatusDot.className = "status-dot offline";
      camStatusText.textContent = "PERMISSION DENIED / UNAVAILABLE";
      let errorMsg = "Unable to access camera: ";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorMsg += "Camera access denied. Please grant permission in browser settings.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        errorMsg += "No webcam detected on your laptop.";
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        errorMsg += "Camera is already in use by another application.";
      } else {
        errorMsg += err.message;
      }
      showToast(errorMsg, "error");
      systemBannerText.textContent = errorMsg;
    }
  }

  function stopCamera() {
    if (streamInstance) {
      streamInstance.getTracks().forEach(track => track.stop());
      streamInstance = null;
    }
    videoFeed.srcObject = null;
    videoFeed.style.display = "none";
    cameraPlaceholder.style.display = "flex";

    btnStartCamera.disabled = false;
    btnStopCamera.disabled = true;
    btnCapture.disabled = true;

    camStatusDot.className = "status-dot offline";
    camStatusText.textContent = "DISCONNECTED";
    systemBannerText.textContent = "Camera stopped. Click Start Camera or browse an image file.";
  }

  // Resume live video view after viewing a captured image
  function resumeStream() {
    if (streamInstance) {
      videoFeed.style.display = "block";
      capturedCanvas.style.display = "none";
      btnResumeStream.style.display = "none";
      btnCapture.disabled = false;
      scanline.classList.remove("active");
      systemBannerText.textContent = "Live webcam stream resumed.";
    } else {
      startCamera();
    }
  }

  // 5. VOICE ALERT SYSTEM (SpeechSynthesis)
  function speakVoiceAlert(message) {
    if (!voiceAlertToggle.checked) return;
    if (!("speechSynthesis" in window)) {
      console.warn("Web Speech API is not supported in this browser.");
      return;
    }

    try {
      window.speechSynthesis.cancel(); // Stop any pending speech
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      // Select natural English voice if available
      const voices = window.speechSynthesis.getVoices();
      const engVoice = voices.find(v => v.lang.startsWith("en") && !v.name.includes("whisper"));
      if (engVoice) utterance.voice = engVoice;

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.error("Audio speech synthesis failed:", e);
    }
  }

  // Test speaker button
  btnTestSpeaker.addEventListener("click", () => {
    speakVoiceAlert("Voice alert system is operational. Warning crack announcement will sound when defect is detected.");
    showToast("Voice alert audio test initiated.", "info");
  });

  // 6. SYSTEM HEALTH CHECK
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.status === "OK") {
        aiStatusDot.className = "status-dot online";
        aiStatusText.textContent = data.ai_models_loaded ? "MODELS READY" : "AI ENGINE ACTIVE";
      } else {
        aiStatusDot.className = "status-dot idle";
        aiStatusText.textContent = "ATTENTION REQUIRED";
      }
    } catch (e) {
      aiStatusDot.className = "status-dot offline";
      aiStatusText.textContent = "BACKEND OFFLINE";
    }
  }

  // 7. IMAGE CAPTURE AND ANALYSIS
  async function captureAndInspect(customBlob = null) {
    if (isInspecting) return;
    isInspecting = true;

    btnCapture.disabled = true;
    btnResumeStream.style.display = "inline-flex";
    scanline.classList.add("active");
    setStepper(1); // Step 1: Capture

    let imageBase64 = null;

    if (customBlob) {
      // From file upload
      imageBase64 = customBlob;
      // Draw onto canvas for preview
      const img = new Image();
      img.onload = () => {
        capturedCanvas.width = img.width;
        capturedCanvas.height = img.height;
        const ctx = capturedCanvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        videoFeed.style.display = "none";
        cameraPlaceholder.style.display = "none";
        capturedCanvas.style.display = "block";
      };
      img.src = customBlob;
    } else {
      // From live video feed
      if (!streamInstance || videoFeed.videoWidth === 0) {
        showToast("Webcam is not streaming active video frames.", "error");
        isInspecting = false;
        btnCapture.disabled = false;
        scanline.classList.remove("active");
        return;
      }

      // Freeze frame onto canvas
      capturedCanvas.width = videoFeed.videoWidth;
      capturedCanvas.height = videoFeed.videoHeight;
      const ctx = capturedCanvas.getContext("2d");
      ctx.drawImage(videoFeed, 0, 0, capturedCanvas.width, capturedCanvas.height);

      videoFeed.style.display = "none";
      capturedCanvas.style.display = "block";
      imageBase64 = capturedCanvas.toDataURL("image/jpeg", 0.95);
    }

    // Step 2: Material Identification
    setStepper(2);
    materialBadge.textContent = "ANALYZING...";
    materialResultText.textContent = "Analyzing Material...";
    materialSubtext.textContent = "Extracting CNN feature maps...";

    try {
      // Send capture to Flask Backend
      const response = await fetch("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageBase64 })
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.error || "Inspection server error");
      }

      // Progress Stepper & Display Results
      displayInspectionResults(result);

    } catch (err) {
      console.error("Inspection error:", err);
      showToast(`Inspection failed: ${err.message}`, "error");
      systemBannerText.textContent = `Inspection error: ${err.message}`;
      materialResultText.textContent = "ERROR";
      materialSubtext.textContent = err.message;
      resetStepper();
    } finally {
      isInspecting = false;
      scanline.classList.remove("active");
      btnCapture.disabled = false;
      refreshStats();
    }
  }

  // 8. DISPLAY INSPECTION RESULTS
  function displayInspectionResults(data) {
    const { material, crack, record } = data;

    // STEP 2 & 3: Material Classification Update
    setStepper(3); // Step 3: Material Gate check
    const matName = material.predicted_class.toUpperCase();
    const matConf = (material.confidence * 100).toFixed(1);

    materialResultText.textContent = matName;
    materialConfidenceLabel.textContent = `${matConf}%`;
    materialMeterFill.style.width = `${matConf}%`;

    // Visual Material Plate styling
    materialIconPlate.className = "material-icon-plate";
    if (matName === "METAL") {
      materialIconPlate.classList.add("metal");
      materialIcon.innerHTML = "&#9874;"; // Anvil/Tools symbol
      materialBadge.textContent = "METAL CONFIRMED";
      materialBadge.className = "card-badge badge-info";
      materialSubtext.textContent = "Metallic surface identified. Passing image to surface crack detector.";
    } else if (matName === "PLASTIC") {
      materialIconPlate.classList.add("plastic");
      materialIcon.innerHTML = "&#9851;"; // Polymer recycle symbol
      materialBadge.textContent = "PLASTIC (SKIPPED)";
      materialBadge.className = "card-badge badge-info";
      materialSubtext.textContent = "Plastic polymer component. Crack inspection is omitted.";
    } else {
      materialIconPlate.classList.add("unknown");
      materialIcon.innerHTML = "?";
      materialBadge.textContent = "UNKNOWN";
      materialBadge.className = "card-badge badge-info";
      materialSubtext.textContent = "Material could not be classified with sufficient confidence.";
    }

    // STEP 4 & 5: Crack Detection & Decision
    setStepper(4);
    setTimeout(() => setStepper(5), 300); // Complete pipeline

    const crackBox = inspectionResultBox;
    crackBox.className = "inspection-result-box";

    if (matName === "METAL") {
      const crackVerdict = crack.predicted_class;
      const crackConf = (crack.confidence * 100).toFixed(1);

      crackConfidenceLabel.textContent = `${crackConf}%`;
      crackMeterFill.style.width = `${crackConf}%`;

      if (crackVerdict === "CRACK DETECTED") {
        crackBox.classList.add("crack-detected");
        resultTag.className = "result-tag defect";
        resultTag.textContent = "CRACK DETECTED";
        resultHeadline.className = "result-headline defect";
        resultHeadline.textContent = "CRACK DETECTED - DEFECTIVE";
        resultDescription.textContent = `A visible surface crack was detected on the metal component (${crackConf}% confidence). Quality status: REJECT.`;
        crackMeterFill.className = "meter-fill defect";
        lblInspectionVerdict.textContent = "REJECT (CRACK)";
        lblInspectionVerdict.style.color = "#ef4444";

        systemBannerText.textContent = "DEFECT DETECTED: Surface crack identified on metal component!";

        // Trigger Voice Announcement (Prevent repeat for identical record ID)
        if (record && record.id !== lastAnnouncedId) {
          lastAnnouncedId = record.id;
          speakVoiceAlert("Warning! Crack detected on the metal surface.");
        }
      } else {
        crackBox.classList.add("no-crack");
        resultTag.className = "result-tag nominal";
        resultTag.textContent = "NO VISIBLE CRACK DETECTED";
        resultHeadline.className = "result-headline nominal";
        resultHeadline.textContent = "SURFACE NOMINAL - PASS";
        resultDescription.textContent = `Metal component surface appears free of detectable cracks (${crackConf}% confidence). Quality status: ACCEPTABLE.`;
        crackMeterFill.className = "meter-fill nominal";
        lblInspectionVerdict.textContent = "PASS (CLEAN)";
        lblInspectionVerdict.style.color = "#10b981";

        systemBannerText.textContent = "INSPECTION PASSED: No visible cracks detected on metal surface.";
      }
    } else if (matName === "PLASTIC") {
      crackBox.classList.add("skipped");
      resultTag.className = "result-tag neutral";
      resultTag.textContent = "INSPECTION NOT PERFORMED";
      resultHeadline.className = "result-headline neutral";
      resultHeadline.textContent = "CRACK INSPECTION SKIPPED";
      resultDescription.textContent = "Plastic component detected. Crack inspection is not performed.";
      crackConfidenceLabel.textContent = "N/A";
      crackMeterFill.style.width = "0%";
      lblInspectionVerdict.textContent = "SKIPPED (PLASTIC)";
      lblInspectionVerdict.style.color = "#94a3b8";

      systemBannerText.textContent = "Plastic component identified. Surface crack inspection is not applicable.";
    } else {
      crackBox.classList.add("skipped");
      resultTag.className = "result-tag neutral";
      resultTag.textContent = "INSPECTION NOT PERFORMED";
      resultHeadline.className = "result-headline neutral";
      resultHeadline.textContent = "INSPECTION ABORTED";
      resultDescription.textContent = "Material not recognised. Inspection cannot be performed.";
      crackConfidenceLabel.textContent = "N/A";
      crackMeterFill.style.width = "0%";
      lblInspectionVerdict.textContent = "UNKNOWN MATERIAL";
      lblInspectionVerdict.style.color = "#94a3b8";

      systemBannerText.textContent = "Material not recognized with sufficient certainty. Inspection cannot proceed.";
    }

    if (record) {
      lblInspectionId.textContent = record.inspection_code || `#${record.id}`;
      lblInspectionTime.textContent = `${record.date_str} ${record.time_str}`;
    }
  }

  // 9. FILE UPLOAD HANDLER (For Testing without webcam)
  manualFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      captureAndInspect(event.target.result);
    };
    reader.readAsDataURL(file);
    // Reset file input so same file can be selected again
    e.target.value = "";
  });

  // 10. INSPECTION HISTORY & DATABASE SYNC
  async function loadHistory() {
    try {
      const res = await fetch("/api/history");
      const records = await res.json();
      historyCounter.textContent = records.length;

      if (!records || records.length === 0) {
        historyTableBody.innerHTML = `
          <tr>
            <td colspan="9" style="text-align: center; color: var(--text-dim); padding: 2.5rem;">
              No inspection records found in SQLite database.
            </td>
          </tr>
        `;
        return;
      }

      historyTableBody.innerHTML = "";
      records.forEach(rec => {
        const tr = document.createElement("tr");

        let crackBadgeClass = "neutral";
        if (rec.crack_result === "CRACK DETECTED") crackBadgeClass = "defect";
        else if (rec.crack_result === "NO VISIBLE CRACK DETECTED") crackBadgeClass = "nominal";

        tr.innerHTML = `
          <td><strong>${rec.inspection_code}</strong></td>
          <td style="font-family: var(--font-mono); font-size: 0.8rem;">${rec.date_str}<br>${rec.time_str}</td>
          <td>
            <img src="/uploads/${rec.image_filename}" class="img-thumb" alt="Sample" 
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2248%22 height=%2248%22><rect width=%2248%22 height=%2248%22 fill=%22%231e293b%22/><text x=%2250%%22 y=%2250%%22 fill=%22%2364748b%22 text-anchor=%22middle%22 dy=%22.3em%22 font-size=%2210%22>No Img</text></svg>'"
                 onclick="openImageModal('/uploads/${rec.image_filename}', '${rec.inspection_code}', '${rec.detected_material} - ${rec.crack_result}')">
          </td>
          <td><span class="card-badge badge-info">${rec.detected_material}</span></td>
          <td>${rec.material_confidence}%</td>
          <td><span class="result-tag ${crackBadgeClass}" style="font-size: 0.7rem; margin:0;">${rec.crack_result}</span></td>
          <td>${rec.crack_confidence !== null ? rec.crack_confidence + '%' : 'N/A'}</td>
          <td><span style="font-size: 0.75rem; color: var(--text-muted);">${rec.status}</span></td>
          <td>
            <button class="btn-icon-del" title="Delete inspection record" onclick="deleteRecord(${rec.id})">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </td>
        `;
        historyTableBody.appendChild(tr);
      });
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }

  async function refreshStats() {
    try {
      const res = await fetch("/api/statistics");
      const stats = await res.json();
      statTotal.textContent = stats.total_inspections;
      statMetal.textContent = stats.metal_inspections;
      statCracks.textContent = stats.cracks_detected;
      const rate = stats.metal_inspections > 0 
        ? ((stats.cracks_detected / stats.metal_inspections) * 100).toFixed(1)
        : "0.0";
      statRate.textContent = `${rate}%`;
      loadHistory();
    } catch (e) {
      console.error("Failed to load statistics:", e);
    }
  }

  // Delete single record
  window.deleteRecord = async (id) => {
    if (!confirm("Are you sure you want to delete this inspection record?")) return;
    try {
      const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
      const data = await res.json();
      if (data.success) {
        showToast("Inspection record removed.", "info");
        refreshStats();
      } else {
        showToast("Failed to delete record.", "error");
      }
    } catch (e) {
      showToast("Delete request failed.", "error");
    }
  };

  // Clear all history
  btnClearHistory.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to CLEAR ALL inspection records? This action cannot be undone.")) return;
    try {
      const res = await fetch("/api/history/clear", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        showToast("All inspection history cleared.", "success");
        refreshStats();
      }
    } catch (e) {
      showToast("Failed to clear history.", "error");
    }
  });

  btnRefreshHistory.addEventListener("click", () => {
    refreshStats();
    showToast("History refreshed.", "info");
  });

  // 11. IMAGE MODAL
  window.openImageModal = (src, code, caption) => {
    modalImg.src = src;
    modalTitle.textContent = `Inspection Inspection Frame: ${code}`;
    modalCaption.textContent = caption;
    imageModal.classList.add("show");
  };

  modalClose.addEventListener("click", () => {
    imageModal.classList.remove("show");
  });

  window.addEventListener("click", (e) => {
    if (e.target === imageModal) imageModal.classList.remove("show");
  });

  // 12. TAB SWITCHING
  tabInspectionBtn.addEventListener("click", () => {
    tabInspectionBtn.classList.add("active");
    tabHistoryBtn.classList.remove("active");
    viewInspection.classList.remove("hidden");
    viewHistory.classList.remove("active");
  });

  tabHistoryBtn.addEventListener("click", () => {
    tabHistoryBtn.classList.add("active");
    tabInspectionBtn.classList.remove("active");
    viewInspection.classList.add("hidden");
    viewHistory.classList.add("active");
    refreshStats();
  });

  // 13. BUTTON EVENT LISTENERS
  btnStartCamera.addEventListener("click", startCamera);
  btnStopCamera.addEventListener("click", stopCamera);
  btnCapture.addEventListener("click", () => captureAndInspect());
  btnResumeStream.addEventListener("click", resumeStream);

  // Initialize
  checkSystemHealth();
  refreshStats();
});

