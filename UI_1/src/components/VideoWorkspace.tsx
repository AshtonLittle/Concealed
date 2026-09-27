import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { ConcealStats } from './Sidebar';

export interface FrameItem {
  frame_index: number;
  sequence_number: number;
  timestamp_sec: number;
  original_image: string;
  obfuscated_image: string;
  difference_image?: string;
  psnr_db: number;
  ssim: number;
  linf_255: number;
  latency_ms: number;
  status: string;
}

export interface VideoMetadata {
  total_video_frames: number;
  processed_frames_count: number;
  fps: number;
  duration_sec: number;
  width: number;
  height: number;
  processed_width?: number;
  processed_height?: number;
  engine?: string;
  backend?: string;
}

export interface VideoAnalytics {
  processing_time_ms: number;
  avg_frame_latency_ms: number;
  avg_fps: number;
  avg_psnr_db: number;
  avg_ssim: number;
}

export type VideoTab = 'concealed' | 'original' | 'side-by-side';

export interface VideoWorkspaceProps {
  onProcessingStart?: () => void;
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const VideoWorkspace: React.FC<VideoWorkspaceProps> = ({ onProcessingStart, onStatsUpdate }) => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // Configuration States
  const [maxFrames, setMaxFrames] = useState<number>(24);
  const [epsilon, setEpsilon] = useState<number>(8.0);
  const [mode, setMode] = useState<string>('hybrid');
  const [autoStartOnDrop, setAutoStartOnDrop] = useState<boolean>(true);

  // Processing & Streaming States
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [targetCount, setTargetCount] = useState<number>(24);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // The Two Videos & Tab State
  const [concealedVideoUrl, setConcealedVideoUrl] = useState<string | null>(null);
  const [activeVideoTab, setActiveVideoTab] = useState<VideoTab>('concealed');
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [analytics, setAnalytics] = useState<VideoAnalytics | null>(null);
  const [engineBackend, setEngineBackend] = useState<string>('ONNX Runtime (generator.onnx)');
  const [frames, setFrames] = useState<FrameItem[]>([]);

  // DOM Refs
  const videoInputRef = useRef<HTMLInputElement>(null);
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const concealedVideoRef = useRef<HTMLVideoElement>(null);

  // Clean up object URLs — separated so revoking one doesn't invalidate the other
  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    };
  }, [videoPreviewUrl]);

  useEffect(() => {
    return () => {
      if (concealedVideoUrl && concealedVideoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(concealedVideoUrl);
      }
    };
  }, [concealedVideoUrl]);

  const hasResults = Boolean((concealedVideoUrl || frames.length > 0) && !isProcessing);

  // Continuous frame-accurate sync between original and concealed in side-by-side mode
  const syncRafRef = useRef<number | null>(null);

  const startSync = useCallback(() => {
    const tick = () => {
      const orig = originalVideoRef.current;
      const conc = concealedVideoRef.current;
      if (orig && conc) {
        // Match playback rate
        if (conc.playbackRate !== orig.playbackRate) {
          conc.playbackRate = orig.playbackRate;
        }
        // Keep currentTime within 0.1s tolerance to avoid constant micro-seeks
        const drift = Math.abs(orig.currentTime - conc.currentTime);
        if (drift > 0.1) {
          conc.currentTime = orig.currentTime;
        }
        // Mirror play/pause state
        if (!orig.paused && conc.paused) {
          conc.play().catch(() => {});
        } else if (orig.paused && !conc.paused) {
          conc.pause();
        }
      }
      syncRafRef.current = requestAnimationFrame(tick);
    };
    // Cancel any existing loop before starting a new one
    if (syncRafRef.current) cancelAnimationFrame(syncRafRef.current);
    syncRafRef.current = requestAnimationFrame(tick);
  }, []);

  const stopSync = useCallback(() => {
    if (syncRafRef.current) {
      cancelAnimationFrame(syncRafRef.current);
      syncRafRef.current = null;
    }
  }, []);

  // Start/stop the sync loop when entering/leaving side-by-side mode
  useEffect(() => {
    if (activeVideoTab === 'side-by-side' && hasResults) {
      startSync();
    } else {
      stopSync();
    }
    return () => stopSync();
  }, [activeVideoTab, hasResults, startSync, stopSync]);

  // Legacy event handlers kept as immediate-response fallbacks for side-by-side
  const handleOriginalPlay = () => {
    if (activeVideoTab === 'side-by-side' && concealedVideoRef.current && concealedVideoRef.current.paused) {
      concealedVideoRef.current.play().catch(() => {});
    }
  };

  const handleOriginalPause = () => {
    if (activeVideoTab === 'side-by-side' && concealedVideoRef.current && !concealedVideoRef.current.paused) {
      concealedVideoRef.current.pause();
    }
  };

  const handleOriginalSeek = () => {
    if (activeVideoTab === 'side-by-side' && originalVideoRef.current && concealedVideoRef.current) {
      concealedVideoRef.current.currentTime = originalVideoRef.current.currentTime;
    }
  };

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processVideoFile = useCallback(async (file: File) => {
    // Notify parent to collapse settings sidebar so screen space is maximized
    if (onProcessingStart) {
      onProcessingStart();
    }

    setSelectedVideo(file);
    const prevUrl = URL.createObjectURL(file);
    setVideoPreviewUrl(prevUrl);
    setConcealedVideoUrl(null);
    setFrames([]);
    setMetadata(null);
    setAnalytics(null);
    setErrorMsg(null);
    setIsProcessing(true);
    setProgressPercent(5);
    setProcessedCount(0);
    setStatusMessage('Decompressing video stream & extracting frames...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', epsilon.toString());
    formData.append('mode', mode);
    formData.append('max_frames', maxFrames.toString());
    formData.append('frame_step', '1');

    try {
      // Step 1: Real-time SSE stream with ONNX acceleration
      setStatusMessage('Initializing ONNX neural generator stream...');
      const streamRes = await fetch('http://127.0.0.1:8001/api/obfuscate/video/stream-frames', {
        method: 'POST',
        body: formData,
      });

      if (streamRes.ok && streamRes.body) {
        const reader = streamRes.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) continue;
            const dataStr = trimmed.slice(5).trim();
            if (!dataStr) continue;

            try {
              const event = JSON.parse(dataStr);

              if (event.type === 'init') {
                const meta = event.metadata;
                setMetadata(meta);
                setTargetCount(meta.frames_to_process || maxFrames);
                if (meta.engine) setEngineBackend(meta.engine);
                setStatusMessage(`Decomposing video • Active Engine: ${meta.engine || 'generator.onnx'}`);
              } else if (event.type === 'frame') {
                const newFrame: FrameItem = event.frame;
                setFrames((prev) => [...prev, newFrame]);
                setProcessedCount(event.processed || 0);
                const pct = Math.round((event.progress || 0) * 100);
                setProgressPercent(Math.min(96, Math.max(10, pct)));
                setStatusMessage(`Applied ONNX obfuscator on Frame #${newFrame.sequence_number} (${newFrame.latency_ms} ms, PSNR ${newFrame.psnr_db} dB)`);
              } else if (event.type === 'complete') {
                const res = event.result;
                if (res.frames && res.frames.length > 0) {
                  setFrames(res.frames);
                }
                if (res.video_base64) {
                  setConcealedVideoUrl(res.video_base64);
                }
                if (res.video_metadata) {
                  setMetadata(res.video_metadata);
                }
                if (res.analytics) {
                  setAnalytics(res.analytics);
                }
                if (res.engine) {
                  setEngineBackend(res.engine);
                }
                setProgressPercent(100);
                setStatusMessage('Video obfuscation complete. Ready for playback.');
                setActiveVideoTab('concealed');
              } else if (event.type === 'error') {
                throw new Error(event.message || 'Stream processing error');
              }
            } catch (jsonErr: unknown) {
              console.warn('SSE chunk skipped:', jsonErr);
            }
          }
        }
      } else {
        // Fallback to standard batch JSON endpoint
        setStatusMessage('Processing video frames via ONNX...');
        const batchRes = await fetch('http://127.0.0.1:8001/api/obfuscate/video/frames', {
          method: 'POST',
          body: formData,
        });

        if (!batchRes.ok) {
          let detail = 'Video obfuscation failed.';
          try {
            const errJson = await batchRes.json();
            detail = errJson.detail || detail;
          } catch {
            detail = `Server HTTP ${batchRes.status}: ${batchRes.statusText}`;
          }
          throw new Error(detail);
        }

        const resData = await batchRes.json();
        setFrames(resData.frames || []);
        if (resData.video_base64) {
          setConcealedVideoUrl(resData.video_base64);
        }
        if (resData.video_metadata) {
          setMetadata(resData.video_metadata);
        }
        if (resData.analytics) {
          setAnalytics(resData.analytics);
        }
        if (resData.engine) {
          setEngineBackend(resData.engine);
        }
        setProgressPercent(100);
        setStatusMessage('Video processing complete.');
        setActiveVideoTab('concealed');
      }
    } catch (err: unknown) {
      console.error('Video obfuscation error:', err);
      const msg = err instanceof Error ? err.message : 'Connection failed';
      setErrorMsg(`${msg}. Verify the Concealed backend is running on http://127.0.0.1:8001.`);
    } finally {
      setIsProcessing(false);
    }
  }, [epsilon, mode, maxFrames, onProcessingStart]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/') || file.name.match(/\.(mp4|webm|mov|avi|mkv)$/i)) {
        if (autoStartOnDrop) {
          processVideoFile(file);
        } else {
          setSelectedVideo(file);
          setVideoPreviewUrl(URL.createObjectURL(file));
          setConcealedVideoUrl(null);
          setFrames([]);
          setErrorMsg(null);
        }
      } else {
        setErrorMsg('Please upload a valid video file (.mp4, .webm, .mov).');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (autoStartOnDrop) {
        processVideoFile(file);
      } else {
        setSelectedVideo(file);
        setVideoPreviewUrl(URL.createObjectURL(file));
        setConcealedVideoUrl(null);
        setFrames([]);
        setErrorMsg(null);
      }
    }
  };

  const triggerVideoInput = () => {
    videoInputRef.current?.click();
  };

  const clearVideo = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedVideo(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(null);
    if (concealedVideoUrl && concealedVideoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(concealedVideoUrl);
    }
    setConcealedVideoUrl(null);
    setFrames([]);
    setMetadata(null);
    setAnalytics(null);
    setErrorMsg(null);
    onStatsUpdate?.(null);
    setIsProcessing(false);
    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  const downloadConcealedVideo = () => {
    if (!concealedVideoUrl) return;
    const a = document.createElement('a');
    a.href = concealedVideoUrl;
    a.download = `concealed_${selectedVideo?.name ? selectedVideo.name.replace(/\.[^/.]+$/, '') : 'video'}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };



  return (
    <main className="video-workspace-screen-wrapper" aria-label="Video protection workspace">
      <div className="video-workspace-screen-container">

        {/* =========================================================================
            STATE 1: INITIAL UPLOAD & SETTINGS (FIT MAJORITY OF SCREEN)
            ========================================================================= */}
        {!hasResults && !isProcessing && (
          <div className="video-upload-screen-section">
            {/* Header info */}
            <div className="video-screen-headline-block">
              <span className="video-badge-pill">
                <span className="video-badge-dot" />
                REAL-TIME ONNX VIDEO OBFUSCATION
              </span>
              <h1 className="video-screen-main-title">Shield Video From Vision Transformers</h1>
              <p className="video-screen-subtitle">
                Decomposes video into individual frames, processes each frame with high-speed quantized ONNX Runtime,
                and outputs protected video undetectable by frontier AI models.
              </p>
            </div>

            {/* Large Wide Video Drop Box */}
            <div
              className={`video-large-dropbox ${isDragging ? 'is-dragging' : ''} ${selectedVideo ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={triggerVideoInput}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  triggerVideoInput();
                }
              }}
              aria-label="Select a video file to obfuscate"
            >
              <input
                type="file"
                ref={videoInputRef}
                onChange={handleFileChange}
                accept="video/*"
                className="dropbox-hidden-input"
                aria-hidden="true"
              />

              {selectedVideo ? (
                <div className="video-selected-file-card" onClick={(e) => e.stopPropagation()}>
                  {videoPreviewUrl && (
                    <video
                      src={videoPreviewUrl}
                      className="video-selected-inline-preview"
                      controls
                    />
                  )}
                  <div className="video-selected-meta">
                    <span className="video-selected-filename">{selectedVideo.name}</span>
                    <span className="video-selected-filesize">
                      {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB • Ready for ONNX Decomposition
                    </span>
                  </div>
                  <button
                    type="button"
                    className="video-selected-remove-btn"
                    onClick={clearVideo}
                    title="Remove selected video"
                    aria-label="Remove selected video"
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="video-dropbox-empty-content">
                  <div className="video-upload-icon-circle">
                    <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="23 7 16 12 23 17 23 7" />
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                    </svg>
                  </div>
                  <span className="video-dropbox-prompt-title">Drop your video file here or click to browse</span>
                  <span className="video-dropbox-prompt-sub">Supports MP4, WebM, MOV, and AVI • Optimized with generator.onnx</span>
                </div>
              )}
            </div>

            {/* Quick Settings Bar */}
            <div className="video-screen-options-bar" role="group" aria-label="Video Frame Extraction Settings">
              <div className="video-screen-option-pill">
                <span className="option-label">FRAME SAMPLE</span>
                <div className="option-btn-segmented">
                  {[12, 24, 48].map((count) => (
                    <button
                      key={count}
                      type="button"
                      className={`opt-chip-btn ${maxFrames === count ? 'active' : ''}`}
                      onClick={() => setMaxFrames(count)}
                    >
                      {count}f
                    </button>
                  ))}
                </div>
              </div>

              <div className="video-screen-option-pill">
                <span className="option-label">BUDGET (ε)</span>
                <div className="option-btn-segmented">
                  {[4, 8, 12].map((eps) => (
                    <button
                      key={eps}
                      type="button"
                      className={`opt-chip-btn ${epsilon === eps ? 'active' : ''}`}
                      onClick={() => setEpsilon(eps)}
                    >
                      {eps}
                    </button>
                  ))}
                </div>
              </div>

              <div className="video-screen-option-pill">
                <span className="option-label">SYNTHESIS</span>
                <div className="option-btn-segmented">
                  {[
                    { id: 'hybrid', label: 'Hybrid' },
                    { id: 'canonical_residual', label: 'Canonical' },
                    { id: 'native', label: 'Native' },
                  ].map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      className={`opt-chip-btn ${mode === m.id ? 'active' : ''}`}
                      onClick={() => setMode(m.id)}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="video-screen-option-pill">
                <label className="option-checkbox-label">
                  <input
                    type="checkbox"
                    checked={autoStartOnDrop}
                    onChange={(e) => setAutoStartOnDrop(e.target.checked)}
                    className="option-checkbox"
                  />
                  <span>Auto-run on Drop</span>
                </label>
              </div>

              <div className="video-screen-option-pill engine-pill">
                <span className="engine-status-dot" />
                <span className="engine-text">ONNX Runtime (Fastest)</span>
              </div>
            </div>

            {/* Large Primary Action Button */}
            <div className="video-screen-action-row">
              <button
                type="button"
                className="video-screen-pushbutton"
                onClick={() => (selectedVideo ? processVideoFile(selectedVideo) : triggerVideoInput())}
                disabled={isProcessing}
                aria-label={selectedVideo ? "Obfuscate video frames using ONNX" : "Select video to obfuscate"}
              >
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="23 7 16 12 23 17 23 7" />
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                </svg>
                <span>{selectedVideo ? 'OBFUSCATE VIDEO FRAMES (ONNX ACCELERATED)' : 'SELECT VIDEO & OBFUSCATE'}</span>
              </button>
            </div>
          </div>
        )}

        {/* =========================================================================
            STATE 2: PROCESSING SCREEN (WIDE-SCREEN RADAR)
            ========================================================================= */}
        {isProcessing && (
          <div className="video-screen-processing-card" role="status" aria-live="polite">
            <div className="video-processing-header">
              <div className="processing-engine-pill">
                <span className="radar-pulse-dot" />
                <span>ONNX RUNTIME ACCELERATED (60 FPS)</span>
              </div>
              <span className="processing-count-label">
                {processedCount} / {targetCount} FRAMES PROCESSED
              </span>
            </div>

            <div className="video-progress-track">
              <div
                className="video-progress-fill"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="video-processing-status-message">
              <div className="video-spinner" />
              <span>{statusMessage}</span>
            </div>

            {/* Live Frame Chips */}
            {frames.length > 0 && (
              <div className="processing-live-frames-wrap">
                <span className="live-frames-title">LIVE ONNX GENERATED FRAMES:</span>
                <div className="live-frames-row">
                  {frames.slice(-8).map((f) => (
                    <div key={f.sequence_number} className="live-frame-mini-card">
                      <img src={f.obfuscated_image} alt={`Frame ${f.sequence_number}`} />
                      <span className="live-frame-seq">#{f.sequence_number}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Alert Box */}
        {errorMsg && (
          <div className="video-screen-error-box" role="alert">
            <span className="error-icon">⚠️</span>
            <span>{errorMsg}</span>
          </div>
        )}

        {/* =========================================================================
            STATE 3: TWO TABS TO SEE THE TWO VIDEOS GENERATED (FIT MAJORITY OF SCREEN)
            ========================================================================= */}
        {hasResults && (
          <div className="video-screen-player-container">
            {/* Top Navigation Bar: The Two Tabs Switcher + File Meta */}
            <div className="video-player-topbar">
              <div className="video-player-file-info">
                <span className="video-meta-filename">{selectedVideo?.name || 'video.mp4'}</span>
                <span className="video-meta-badge">{metadata?.processed_frames_count || frames.length} FRAMES</span>
                <span className="video-meta-badge engine-badge">{engineBackend}</span>
              </div>

              {/* THE TWO TABS */}
              <div className="video-two-tabs-segmented" role="tablist" aria-label="Video View Switcher">
                {/* TAB 1: CONCEALED VIDEO */}
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeVideoTab === 'concealed'}
                  className={`video-segmented-tab ${activeVideoTab === 'concealed' ? 'active' : ''}`}
                  onClick={() => setActiveVideoTab('concealed')}
                  title="View the generated concealed video with adversarial perturbations"
                >
                  <span className="tab-icon">🛡️</span>
                  <span>CONCEALED VIDEO</span>
                </button>

                {/* TAB 2: ORIGINAL VIDEO */}
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeVideoTab === 'original'}
                  className={`video-segmented-tab ${activeVideoTab === 'original' ? 'active' : ''}`}
                  onClick={() => setActiveVideoTab('original')}
                  title="View the original unmodified input video"
                >
                  <span className="tab-icon">📹</span>
                  <span>ORIGINAL VIDEO</span>
                </button>

                {/* SIDE-BY-SIDE TAB */}
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeVideoTab === 'side-by-side'}
                  className={`video-segmented-tab ${activeVideoTab === 'side-by-side' ? 'active' : ''}`}
                  onClick={() => setActiveVideoTab('side-by-side')}
                  title="Compare both videos side-by-side simultaneously"
                >
                  <span className="tab-icon">◫</span>
                  <span>SIDE-BY-SIDE</span>
                </button>
              </div>

              {/* Reset Button */}
              <button
                type="button"
                className="video-reset-button"
                onClick={clearVideo}
                title="Process another video"
              >
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2">
                  <path d="M23 4v6h-6" />
                  <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                </svg>
                <span>NEW VIDEO</span>
              </button>
            </div>

            {/* =====================================================================
                CINEMA-SCALE VIDEO STAGE (FITS MAJORITY OF SCREEN)
                ===================================================================== */}
            <div className="video-cinema-viewport-stage">
              {/* TAB 1: CONCEALED VIDEO ONLY */}
              {activeVideoTab === 'concealed' && (
                <div className="single-video-viewport">
                  <div className="video-overlay-pill concealed">
                    <span className="pill-dot purple" />
                    <span>CONCEALED ADVERSARIAL VIDEO (ONNX ACCELERATED)</span>
                  </div>
                  {concealedVideoUrl ? (
                    <video
                      key="concealed-player"
                      src={concealedVideoUrl}
                      controls
                      autoPlay
                      loop
                      className="cinema-video-element"
                    />
                  ) : (
                    <video
                      key="preview-fallback"
                      src={videoPreviewUrl || ''}
                      controls
                      autoPlay
                      loop
                      className="cinema-video-element"
                    />
                  )}
                </div>
              )}

              {/* TAB 2: ORIGINAL VIDEO ONLY */}
              {activeVideoTab === 'original' && (
                <div className="single-video-viewport">
                  <div className="video-overlay-pill original">
                    <span className="pill-dot blue" />
                    <span>ORIGINAL UNMODIFIED SOURCE VIDEO</span>
                  </div>
                  <video
                    key="original-player"
                    src={videoPreviewUrl || ''}
                    controls
                    autoPlay
                    loop
                    className="cinema-video-element"
                  />
                </div>
              )}

              {/* TAB 3: SIDE-BY-SIDE DUAL VIEW */}
              {activeVideoTab === 'side-by-side' && (
                <div className="side-by-side-dual-viewport">
                  <div className="dual-video-pane">
                    <div className="video-overlay-pill original">
                      <span className="pill-dot blue" />
                      <span>ORIGINAL VIDEO</span>
                    </div>
                    <video
                      ref={originalVideoRef}
                      src={videoPreviewUrl || ''}
                      controls
                      autoPlay
                      loop
                      onPlay={handleOriginalPlay}
                      onPause={handleOriginalPause}
                      onSeeked={handleOriginalSeek}
                      className="cinema-video-element dual"
                    />
                  </div>

                  <div className="dual-video-pane">
                    <div className="video-overlay-pill concealed">
                      <span className="pill-dot purple" />
                      <span>CONCEALED VIDEO (ONNX)</span>
                    </div>
                    {concealedVideoUrl ? (
                      <video
                        ref={concealedVideoRef}
                        src={concealedVideoUrl}
                        controls
                        autoPlay
                        loop
                        muted
                        className="cinema-video-element dual"
                      />
                    ) : (
                      <video
                        ref={concealedVideoRef}
                        src={videoPreviewUrl || ''}
                        controls
                        autoPlay
                        loop
                        muted
                        className="cinema-video-element dual"
                      />
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Telemetry HUD & Download Action */}
            <div className="video-cinema-footer-bar">
              <div className="video-telemetry-chips">
                <div className="telemetry-chip">
                  <span className="chip-key">FRAMES</span>
                  <span className="chip-val highlight">{metadata?.processed_frames_count || frames.length}</span>
                </div>
                <div className="telemetry-chip">
                  <span className="chip-key">FPS</span>
                  <span className="chip-val">{metadata?.fps || 24}</span>
                </div>
                {analytics && (
                  <>
                    <div className="telemetry-chip">
                      <span className="chip-key">AVG LATENCY</span>
                      <span className="chip-val">{analytics.avg_frame_latency_ms.toFixed(1)} ms</span>
                    </div>
                    <div className="telemetry-chip">
                      <span className="chip-key">AVG PSNR</span>
                      <span className="chip-val highlight">{analytics.avg_psnr_db.toFixed(1)} dB</span>
                    </div>
                    <div className="telemetry-chip">
                      <span className="chip-key">AVG SSIM</span>
                      <span className="chip-val">{analytics.avg_ssim.toFixed(4)}</span>
                    </div>
                  </>
                )}
                <div className="telemetry-chip">
                  <span className="chip-key">ENGINE</span>
                  <span className="chip-val highlight">{engineBackend}</span>
                </div>
              </div>

              {/* Download Concealed Video Button */}
              {concealedVideoUrl && (
                <button
                  type="button"
                  className="video-download-action-btn"
                  onClick={downloadConcealedVideo}
                  title="Download the full shielded video as MP4"
                >
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Download Concealed Video (.mp4)</span>
                </button>
              )}
            </div>
          </div>
        )}

      </div>
    </main>
  );
};
