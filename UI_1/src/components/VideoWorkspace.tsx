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

export type ViewportMode = 'slider' | 'side-by-side' | 'concealed' | 'original' | 'difference';
export type WorkspaceTab = 'visualizer' | 'video-player';

export interface VideoWorkspaceProps {
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const VideoWorkspace: React.FC<VideoWorkspaceProps> = ({ onStatsUpdate }) => {
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

  // Results & Frame Visualizer States
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>('visualizer');
  const [viewportMode, setViewportMode] = useState<ViewportMode>('slider');
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState<boolean>(false);

  const [frames, setFrames] = useState<FrameItem[]>([]);
  const [activeFrameIdx, setActiveFrameIdx] = useState<number>(0);
  const [isPlayingSlideshow, setIsPlayingSlideshow] = useState<boolean>(false);
  const [slideshowFps, setSlideshowFps] = useState<number>(4);

  // Reconstructed Video Player States
  const [concealedVideoUrl, setConcealedVideoUrl] = useState<string | null>(null);
  const [videoPlayerSource, setVideoPlayerSource] = useState<'concealed' | 'original'>('concealed');
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [analytics, setAnalytics] = useState<VideoAnalytics | null>(null);
  const [engineBackend, setEngineBackend] = useState<string>('ONNXRuntime (generator.onnx)');

  // DOM Refs
  const videoInputRef = useRef<HTMLInputElement>(null);
  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const filmstripRef = useRef<HTMLDivElement>(null);
  const slideshowTimerRef = useRef<number | null>(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      if (concealedVideoUrl) URL.revokeObjectURL(concealedVideoUrl);
    };
  }, [videoPreviewUrl, concealedVideoUrl]);

  // Slideshow auto-play effect
  useEffect(() => {
    if (isPlayingSlideshow && frames.length > 1) {
      const interval = Math.max(40, 1000 / slideshowFps);
      slideshowTimerRef.current = window.setInterval(() => {
        setActiveFrameIdx((prev) => (prev + 1) % frames.length);
      }, interval);
    } else {
      if (slideshowTimerRef.current) {
        clearInterval(slideshowTimerRef.current);
        slideshowTimerRef.current = null;
      }
    }
    return () => {
      if (slideshowTimerRef.current) {
        clearInterval(slideshowTimerRef.current);
      }
    };
  }, [isPlayingSlideshow, frames.length, slideshowFps]);

  // Auto-scroll active thumbnail into view
  useEffect(() => {
    if (filmstripRef.current && frames.length > 0) {
      const activeEl = filmstripRef.current.children[activeFrameIdx] as HTMLElement;
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      }
    }
  }, [activeFrameIdx, frames.length]);

  // Keyboard navigation for frames
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (frames.length === 0 || isProcessing) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setActiveFrameIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setActiveFrameIdx((prev) => Math.min(frames.length - 1, prev + 1));
      } else if (e.key === ' ') {
        e.preventDefault();
        setIsPlayingSlideshow((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [frames.length, isProcessing]);

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processVideoFile = useCallback(async (file: File) => {
    setSelectedVideo(file);
    const prevUrl = URL.createObjectURL(file);
    setVideoPreviewUrl(prevUrl);
    setConcealedVideoUrl(null);
    setFrames([]);
    setActiveFrameIdx(0);
    setMetadata(null);
    setAnalytics(null);
    setErrorMsg(null);
    setIsProcessing(true);
    setProgressPercent(5);
    setProcessedCount(0);
    setStatusMessage('Decompressing video stream & extracting keyframes...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', epsilon.toString());
    formData.append('mode', mode);
    formData.append('max_frames', maxFrames.toString());
    formData.append('frame_step', '1');

    try {
      // Step 1: Attempt real-time SSE stream
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
                setStatusMessage(`Decomposing video into frames • Active ONNX Model: ${meta.engine || 'generator.onnx'}`);
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
                setStatusMessage('All video frames shielded & reconstructed with ONNX.');
              } else if (event.type === 'error') {
                throw new Error(event.message || 'Stream processing error');
              }
            } catch (jsonErr: unknown) {
              console.warn('SSE JSON parse chunk skip:', jsonErr);
            }
          }
        }
      } else {
        // Fallback to standard batch JSON endpoint
        setStatusMessage('Streaming fallback: Processing full frame sequence via fast ONNX...');
        const batchRes = await fetch('http://127.0.0.1:8001/api/obfuscate/video/frames', {
          method: 'POST',
          body: formData,
        });

        if (!batchRes.ok) {
          let detail = 'Video frame obfuscation failed.';
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
        setStatusMessage('Video frame processing complete.');
      }
    } catch (err: unknown) {
      console.error('Video obfuscation error:', err);
      const msg = err instanceof Error ? err.message : 'Connection failed';
      setErrorMsg(`${msg}. Verify the Concealed backend is running on http://127.0.0.1:8001.`);
    } finally {
      setIsProcessing(false);
    }
  }, [epsilon, mode, maxFrames]);

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
          setMetadata(null);
          setAnalytics(null);
          setErrorMsg(null);
        }
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
        setMetadata(null);
        setAnalytics(null);
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
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
      setVideoPreviewUrl(null);
    }
    if (concealedVideoUrl) {
      URL.revokeObjectURL(concealedVideoUrl);
      setConcealedVideoUrl(null);
    }
    setFrames([]);
    setActiveFrameIdx(0);
    setMetadata(null);
    setAnalytics(null);
    setErrorMsg(null);
    onStatsUpdate?.(null);
    setIsProcessing(false);
    setIsPlayingSlideshow(false);
    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  // Slider Mouse/Touch Drag Handlers
  const handleSliderMove = useCallback((clientX: number) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(Math.round(pct));
  }, []);

  const handleMouseDown = () => setIsDraggingSlider(true);
  const handleTouchStart = () => setIsDraggingSlider(true);

  useEffect(() => {
    const handleMouseUp = () => setIsDraggingSlider(false);
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingSlider) handleSliderMove(e.clientX);
    };
    const handleTouchMove = (e: TouchEvent) => {
      if (isDraggingSlider && e.touches.length > 0) {
        handleSliderMove(e.touches[0].clientX);
      }
    };

    if (isDraggingSlider) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDraggingSlider, handleSliderMove]);

  // Download Handlers
  const downloadConcealedVideo = () => {
    if (!concealedVideoUrl) return;
    const a = document.createElement('a');
    a.href = concealedVideoUrl;
    a.download = `concealed_${selectedVideo?.name || 'video.mp4'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const downloadActiveFrame = () => {
    const frame = frames[activeFrameIdx];
    if (!frame) return;
    const a = document.createElement('a');
    a.href = frame.obfuscated_image;
    a.download = `frame_${frame.sequence_number}_obfuscated.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const downloadDifferenceMap = () => {
    const frame = frames[activeFrameIdx];
    if (!frame || !frame.difference_image) return;
    const a = document.createElement('a');
    a.href = frame.difference_image;
    a.download = `frame_${frame.sequence_number}_perturbation_map.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const activeFrame: FrameItem | undefined = frames[activeFrameIdx];

  return (
    <main className="main-workspace video-workspace-root" aria-label="Video protection workspace">
      <div className="workspace-content">
        <div className="protection-status-center" aria-live="polite">

          {/* Top Video Drop Box (Visible when not yet processing, or minimal bar when results exist) */}
          {!frames.length && !isProcessing && (
            <>
              <div
                className={`image-dropbox video-dropbox ${isDragging ? 'is-dragging' : ''} ${selectedVideo ? 'has-file' : ''}`}
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
                aria-label="Select a video file to break into frames and obfuscate"
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
                  <div className="dropbox-file-info video-file-info">
                    {videoPreviewUrl ? (
                      <video
                        src={videoPreviewUrl}
                        className="video-inline-preview"
                        controls
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <svg
                        className="dropbox-file-icon"
                        viewBox="0 0 24 24"
                        width="36"
                        height="36"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                        <line x1="7" y1="2" x2="7" y2="22" />
                        <line x1="17" y1="2" x2="17" y2="22" />
                        <line x1="2" y1="12" x2="22" y2="12" />
                        <line x1="2" y1="7" x2="7" y2="7" />
                        <line x1="2" y1="17" x2="7" y2="17" />
                      </svg>
                    )}
                    <div className="dropbox-file-details">
                      <span className="dropbox-filename">{selectedVideo.name}</span>
                      <span className="dropbox-filesize">
                        {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB • Ready for Frame Decomposition
                      </span>
                    </div>
                    <button
                      type="button"
                      className="dropbox-clear-btn"
                      onClick={clearVideo}
                      title="Remove selected video"
                      aria-label="Remove selected video"
                    >
                      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                ) : (
                  <div className="dropbox-content">
                    <svg
                      className="dropbox-upload-icon"
                      viewBox="0 0 24 24"
                      width="44"
                      height="44"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <polygon points="23 7 16 12 23 17 23 7" />
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                    </svg>
                    <span className="dropbox-prompt-title">Select or drop a video file</span>
                    <span className="dropbox-prompt-subtitle">Breaks video into frames & shields with high-speed ONNX model</span>
                  </div>
                )}
              </div>

              {/* Fast Settings Controls before processing */}
              <div className="video-options-bar" role="group" aria-label="Video Frame Extraction Settings">
                <div className="video-option-pill">
                  <span className="option-pill-label">FRAME SAMPLE</span>
                  <div className="option-segmented">
                    {[12, 24, 48].map((count) => (
                      <button
                        key={count}
                        type="button"
                        className={`opt-btn ${maxFrames === count ? 'active' : ''}`}
                        onClick={() => setMaxFrames(count)}
                      >
                        {count}f
                      </button>
                    ))}
                  </div>
                </div>

                <div className="video-option-pill">
                  <span className="option-pill-label">BUDGET ε</span>
                  <div className="option-segmented">
                    {[4, 8, 12].map((eps) => (
                      <button
                        key={eps}
                        type="button"
                        className={`opt-btn ${epsilon === eps ? 'active' : ''}`}
                        onClick={() => setEpsilon(eps)}
                      >
                        {eps}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="video-option-pill">
                  <span className="option-pill-label">SYNTHESIS</span>
                  <div className="option-segmented">
                    {[
                      { id: 'hybrid', label: 'Hybrid' },
                      { id: 'canonical_residual', label: 'Canonical' },
                      { id: 'native', label: 'Native' },
                    ].map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        className={`opt-btn ${mode === m.id ? 'active' : ''}`}
                        onClick={() => setMode(m.id)}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="video-option-pill">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', fontSize: '11px', color: '#c0c0cb' }}>
                    <input
                      type="checkbox"
                      checked={autoStartOnDrop}
                      onChange={(e) => setAutoStartOnDrop(e.target.checked)}
                      style={{ accentColor: '#00f5a0', cursor: 'pointer' }}
                    />
                    <span>Auto-run on Drop</span>
                  </label>
                </div>

                <div className="video-option-pill">
                  <span className="option-pill-label">ENGINE</span>
                  <span className="engine-chip-pill">ONNX Runtime</span>
                </div>
              </div>

              {/* Primary Action Button */}
              <div className="workspace-action-row" style={{ marginTop: '14px' }}>
                <button
                  type="button"
                  className="main-action-pushbutton"
                  onClick={() => selectedVideo ? processVideoFile(selectedVideo) : triggerVideoInput()}
                  disabled={isProcessing}
                  aria-label={selectedVideo ? "Obfuscate video frames using ONNX" : "Select video to obfuscate"}
                >
                  <span className="pushbutton-content">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="23 7 16 12 23 17 23 7" />
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                    </svg>
                    <span>{selectedVideo ? 'OBFUSCATE VIDEO FRAMES (ONNX ACCELERATED)' : 'SELECT VIDEO & OBFUSCATE'}</span>
                  </span>
                </button>
              </div>

              <h2 className="protection-headline">Stay Concealed</h2>
              <p className="protection-tagline">Real-Time Frame Obfuscation • ONNX Engine</p>
            </>
          )}

          {/* Processing Screen with Live Progress */}
          {isProcessing && (
            <div className="video-processing-card" role="status" aria-live="polite">
              <div className="processing-header-row">
                <div className="processing-engine-badge">
                  <span className="radar-dot" />
                  <span>ONNX RUNTIME ACCELERATED</span>
                </div>
                <span className="processing-frame-counter">
                  {processedCount} / {targetCount} FRAMES
                </span>
              </div>

              <div className="processing-progress-track">
                <div
                  className="processing-progress-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>

              <div className="processing-status-text">
                <div className="spinner-dots" />
                <span>{statusMessage}</span>
              </div>

              {/* Live Preview of Last Processed Frame while processing */}
              {frames.length > 0 && (
                <div className="processing-live-frame-strip">
                  <span className="live-frame-label">LIVE STREAMING OBFUSCATED FRAMES:</span>
                  <div className="live-frame-row">
                    {frames.slice(-6).map((f) => (
                      <div key={f.sequence_number} className="live-frame-chip">
                        <img src={f.obfuscated_image} alt={`Frame ${f.sequence_number}`} />
                        <span className="live-frame-num">#{f.sequence_number}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error Alert Box */}
          {errorMsg && (
            <div className="feature-error-box" style={{ maxWidth: '640px', marginBottom: '16px' }} role="alert">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Completed Video & Frame Visualizer Workspace */}
          {frames.length > 0 && !isProcessing && (
            <div className="video-visualizer-container">

              {/* Workspace Top Toolbar: View Switcher & File Details */}
              <div className="visualizer-header-toolbar">
                <div className="visualizer-file-meta">
                  <span className="meta-filename">{selectedVideo?.name || 'video.mp4'}</span>
                  <span className="meta-badge-count">{frames.length} FRAMES SHIELDED</span>
                  <span className="meta-engine-tag">{engineBackend}</span>
                </div>

                <div className="visualizer-tab-toggle" role="tablist">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={workspaceTab === 'visualizer'}
                    className={`viz-tab-btn ${workspaceTab === 'visualizer' ? 'active' : ''}`}
                    onClick={() => setWorkspaceTab('visualizer')}
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <line x1="3" y1="9" x2="21" y2="9" />
                      <line x1="9" y1="21" x2="9" y2="9" />
                    </svg>
                    <span>FRAME VISUALIZER</span>
                  </button>

                  <button
                    type="button"
                    role="tab"
                    aria-selected={workspaceTab === 'video-player'}
                    className={`viz-tab-btn ${workspaceTab === 'video-player' ? 'active' : ''}`}
                    onClick={() => setWorkspaceTab('video-player')}
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    <span>RECONSTRUCTED VIDEO</span>
                  </button>

                  <button
                    type="button"
                    className="viz-tab-btn new-video-btn"
                    onClick={clearVideo}
                    title="Process another video"
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    <span>NEW VIDEO</span>
                  </button>
                </div>
              </div>

              {/* TAB 1: FRAME VISUALIZER */}
              {workspaceTab === 'visualizer' && activeFrame && (
                <div className="frame-visualizer-main-panel">

                  {/* Visualizer Viewport Mode Switcher */}
                  <div className="visualizer-mode-bar">
                    <span className="mode-bar-label">INSPECTOR MODE:</span>
                    <div className="mode-btn-group">
                      <button
                        type="button"
                        className={`sub-mode-btn ${viewportMode === 'slider' ? 'active' : ''}`}
                        onClick={() => setViewportMode('slider')}
                        title="Interactive Before / After Wipe Slider"
                      >
                        Split Slider
                      </button>
                      <button
                        type="button"
                        className={`sub-mode-btn ${viewportMode === 'side-by-side' ? 'active' : ''}`}
                        onClick={() => setViewportMode('side-by-side')}
                        title="Side-by-side comparison"
                      >
                        Side-by-Side
                      </button>
                      <button
                        type="button"
                        className={`sub-mode-btn ${viewportMode === 'concealed' ? 'active' : ''}`}
                        onClick={() => setViewportMode('concealed')}
                        title="Show full shielded frame"
                      >
                        Shielded Only
                      </button>
                      <button
                        type="button"
                        className={`sub-mode-btn ${viewportMode === 'original' ? 'active' : ''}`}
                        onClick={() => setViewportMode('original')}
                        title="Show original frame"
                      >
                        Original
                      </button>
                      {activeFrame.difference_image && (
                        <button
                          type="button"
                          className={`sub-mode-btn ${viewportMode === 'difference' ? 'active' : ''}`}
                          onClick={() => setViewportMode('difference')}
                          title="Show adversarial perturbation noise heatmap"
                        >
                          Perturbation Map
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Active Frame Display Viewport */}
                  <div className="frame-viewport-stage">
                    {viewportMode === 'slider' && (
                      <div
                        ref={sliderContainerRef}
                        className="interactive-slider-wrapper"
                        onMouseDown={handleMouseDown}
                        onTouchStart={handleTouchStart}
                        style={{ cursor: isDraggingSlider ? 'ew-resize' : 'default' }}
                      >
                        {/* Background: Original Image */}
                        <img
                          src={activeFrame.original_image}
                          alt={`Frame ${activeFrame.sequence_number} Original`}
                          className="slider-base-img"
                          draggable={false}
                        />
                        <span className="slider-label slider-label-left">ORIGINAL</span>

                        {/* Foreground: Obfuscated Image (Clipped) */}
                        <div
                          className="slider-overlay"
                          style={{
                            clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)`,
                          }}
                        >
                          <img
                            src={activeFrame.obfuscated_image}
                            alt={`Frame ${activeFrame.sequence_number} Shielded`}
                            className="slider-overlay-img"
                            draggable={false}
                          />
                          <span className="slider-label slider-label-right">SHIELDED (ONNX)</span>
                        </div>

                        {/* Draggable Divider Line & Knob */}
                        <div
                          className="slider-divider"
                          style={{ left: `${sliderPos}%` }}
                        >
                          <div className="slider-handle">
                            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5">
                              <polyline points="15 18 9 12 15 6" />
                              <polyline points="9 18 15 12 9 6" />
                            </svg>
                          </div>
                        </div>
                      </div>
                    )}

                    {viewportMode === 'side-by-side' && (
                      <div className="side-by-side-wrapper">
                        <div className="side-frame-card">
                          <div className="side-frame-header">
                            <span className="side-tag original">ORIGINAL FRAME #{activeFrame.sequence_number}</span>
                          </div>
                          <img
                            src={activeFrame.original_image}
                            alt="Original frame"
                            className="side-frame-img"
                          />
                        </div>

                        <div className="side-frame-card">
                          <div className="side-frame-header">
                            <span className="side-tag shielded">SHIELDED (ONNX)</span>
                            <span className="side-metric">SSIM: {activeFrame.ssim}</span>
                          </div>
                          <img
                            src={activeFrame.obfuscated_image}
                            alt="Obfuscated frame"
                            className="side-frame-img"
                          />
                        </div>
                      </div>
                    )}

                    {viewportMode === 'concealed' && (
                      <div className="single-frame-wrapper">
                        <img
                          src={activeFrame.obfuscated_image}
                          alt="Concealed frame"
                          className="single-frame-img"
                        />
                        <span className="viewport-overlay-badge shielded">SHIELDED • {activeFrame.psnr_db} dB PSNR</span>
                      </div>
                    )}

                    {viewportMode === 'original' && (
                      <div className="single-frame-wrapper">
                        <img
                          src={activeFrame.original_image}
                          alt="Original frame"
                          className="single-frame-img"
                        />
                        <span className="viewport-overlay-badge original">UNMODIFIED ORIGINAL</span>
                      </div>
                    )}

                    {viewportMode === 'difference' && activeFrame.difference_image && (
                      <div className="single-frame-wrapper">
                        <img
                          src={activeFrame.difference_image}
                          alt="Adversarial noise delta heatmap"
                          className="single-frame-img"
                        />
                        <span className="viewport-overlay-badge diff">ADVERSARIAL PERTURBATION MAP (6x AMPLIFIED)</span>
                      </div>
                    )}
                  </div>

                  {/* Active Frame Status & Telemetry HUD */}
                  <div className="frame-telemetry-hud">
                    <div className="hud-badge-group">
                      <div className="hud-badge status-badge">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                        </svg>
                        <span>100% SHIELDED</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">FRAME</span>
                        <span className="hud-v highlight">#{activeFrame.sequence_number} / {frames.length}</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">TIMESTAMP</span>
                        <span className="hud-v">{activeFrame.timestamp_sec.toFixed(2)}s</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">INFERENCE</span>
                        <span className="hud-v">{activeFrame.latency_ms.toFixed(1)} ms</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">PSNR</span>
                        <span className="hud-v highlight">{activeFrame.psnr_db} dB</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">SSIM</span>
                        <span className="hud-v">{activeFrame.ssim}</span>
                      </div>

                      <div className="hud-badge">
                        <span className="hud-k">L_INF</span>
                        <span className="hud-v">{activeFrame.linf_255} / 255</span>
                      </div>
                    </div>

                    <div className="hud-actions-group">
                      <button
                        type="button"
                        className="hud-action-btn"
                        onClick={downloadActiveFrame}
                        title="Download this shielded frame"
                      >
                        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        <span>Frame</span>
                      </button>

                      {activeFrame.difference_image && (
                        <button
                          type="button"
                          className="hud-action-btn"
                          onClick={downloadDifferenceMap}
                          title="Download adversarial noise mask"
                        >
                          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" />
                            <circle cx="12" cy="12" r="3" />
                          </svg>
                          <span>Perturbation</span>
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Playback Controls & Frame Scrubber */}
                  <div className="frame-playback-controls">
                    <button
                      type="button"
                      className="nav-step-btn"
                      onClick={() => setActiveFrameIdx(0)}
                      disabled={activeFrameIdx === 0}
                      title="Jump to first frame"
                    >
                      |&lt;
                    </button>

                    <button
                      type="button"
                      className="nav-step-btn"
                      onClick={() => setActiveFrameIdx((prev) => Math.max(0, prev - 1))}
                      disabled={activeFrameIdx === 0}
                      title="Previous frame (Left Arrow)"
                    >
                      &lt; Prev
                    </button>

                    <button
                      type="button"
                      className={`play-slideshow-btn ${isPlayingSlideshow ? 'playing' : ''}`}
                      onClick={() => setIsPlayingSlideshow((prev) => !prev)}
                      title={isPlayingSlideshow ? 'Pause animation (Space)' : 'Play animation (Space)'}
                    >
                      {isPlayingSlideshow ? (
                        <>
                          <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                            <rect x="6" y="4" width="4" height="16" />
                            <rect x="14" y="4" width="4" height="16" />
                          </svg>
                          <span>PAUSE</span>
                        </>
                      ) : (
                        <>
                          <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          <span>PLAY SLIDESHOW</span>
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      className="nav-step-btn"
                      onClick={() => setActiveFrameIdx((prev) => Math.min(frames.length - 1, prev + 1))}
                      disabled={activeFrameIdx === frames.length - 1}
                      title="Next frame (Right Arrow)"
                    >
                      Next &gt;
                    </button>

                    <button
                      type="button"
                      className="nav-step-btn"
                      onClick={() => setActiveFrameIdx(frames.length - 1)}
                      disabled={activeFrameIdx === frames.length - 1}
                      title="Jump to last frame"
                    >
                      &gt;|
                    </button>

                    {/* Timeline Scrubber */}
                    <div className="scrubber-range-container">
                      <input
                        type="range"
                        min={0}
                        max={frames.length - 1}
                        value={activeFrameIdx}
                        onChange={(e) => setActiveFrameIdx(parseInt(e.target.value, 10))}
                        className="timeline-scrubber"
                      />
                    </div>

                    {/* FPS Speed Selector */}
                    <div className="slideshow-fps-picker">
                      <span className="fps-label">FPS:</span>
                      {[2, 4, 8, 16].map((fpsVal) => (
                        <button
                          key={fpsVal}
                          type="button"
                          className={`fps-chip ${slideshowFps === fpsVal ? 'active' : ''}`}
                          onClick={() => setSlideshowFps(fpsVal)}
                        >
                          {fpsVal}x
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Horizontal Interactive Filmstrip Carousel */}
                  <div className="filmstrip-container">
                    <div className="filmstrip-title-row">
                      <span className="filmstrip-title">FRAME SEQUENCE FILMSTRIP ({frames.length} EXTRACTED FRAMES)</span>
                      <span className="filmstrip-hint">Click thumbnail to inspect</span>
                    </div>

                    <div ref={filmstripRef} className="filmstrip-scroll">
                      {frames.map((f, idx) => (
                        <div
                          key={f.sequence_number}
                          className={`filmstrip-card ${idx === activeFrameIdx ? 'active' : ''}`}
                          onClick={() => setActiveFrameIdx(idx)}
                          role="button"
                          tabIndex={0}
                          aria-label={`Frame ${f.sequence_number} at ${f.timestamp_sec}s`}
                        >
                          <div className="filmstrip-thumb-box">
                            <img
                              src={f.obfuscated_image}
                              alt={`Frame ${f.sequence_number}`}
                              className="filmstrip-thumb-img"
                              loading="lazy"
                            />
                            <span className="filmstrip-badge">#{f.sequence_number}</span>
                            <span className="filmstrip-shield-dot" title="Shielded with ONNX" />
                          </div>
                          <div className="filmstrip-card-footer">
                            <span className="filmstrip-time">{f.timestamp_sec.toFixed(2)}s</span>
                            <span className="filmstrip-psnr">{f.psnr_db}dB</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              )}

              {/* TAB 2: RECONSTRUCTED VIDEO PLAYER */}
              {workspaceTab === 'video-player' && (
                <div className="reconstructed-video-panel">
                  {/* Video Source Comparison Switcher */}
                  <div className="video-source-switcher">
                    <button
                      type="button"
                      className={`source-btn ${videoPlayerSource === 'concealed' ? 'active' : ''}`}
                      onClick={() => setVideoPlayerSource('concealed')}
                    >
                      <span className="dot dot-green" />
                      <span>Shielded Reconstructed Video</span>
                    </button>
                    <button
                      type="button"
                      className={`source-btn ${videoPlayerSource === 'original' ? 'active' : ''}`}
                      onClick={() => setVideoPlayerSource('original')}
                    >
                      <span className="dot dot-gray" />
                      <span>Original Video</span>
                    </button>
                  </div>

                  {/* Main Video Viewport */}
                  <div className="reconstructed-video-stage">
                    {videoPlayerSource === 'concealed' && concealedVideoUrl ? (
                      <video
                        src={concealedVideoUrl}
                        controls
                        autoPlay
                        loop
                        className="main-reconstructed-video"
                      />
                    ) : (
                      <video
                        src={videoPreviewUrl || ''}
                        controls
                        autoPlay
                        loop
                        className="main-reconstructed-video"
                      />
                    )}
                  </div>

                  {/* Summary Analytics Badges */}
                  <div className="telemetry-badges-row" style={{ justifyContent: 'center', marginTop: '16px' }}>
                    <div className="telemetry-badge">
                      <span className="badge-k">FRAMES</span>
                      <span className="badge-v highlight">{metadata?.processed_frames_count || frames.length}</span>
                    </div>
                    <div className="telemetry-badge">
                      <span className="badge-k">ORIGINAL FPS</span>
                      <span className="badge-v">{metadata?.fps || 24}</span>
                    </div>
                    {analytics && (
                      <>
                        <div className="telemetry-badge">
                          <span className="badge-k">AVG LATENCY</span>
                          <span className="badge-v">{analytics.avg_frame_latency_ms.toFixed(1)} ms</span>
                        </div>
                        <div className="telemetry-badge">
                          <span className="badge-k">AVG PSNR</span>
                          <span className="badge-v highlight">{analytics.avg_psnr_db.toFixed(1)} dB</span>
                        </div>
                        <div className="telemetry-badge">
                          <span className="badge-k">AVG SSIM</span>
                          <span className="badge-v">{analytics.avg_ssim.toFixed(4)}</span>
                        </div>
                      </>
                    )}
                    <div className="telemetry-badge">
                      <span className="badge-k">ENGINE</span>
                      <span className="badge-v highlight">{engineBackend}</span>
                    </div>
                  </div>

                  {/* Video Actions */}
                  <div className="video-actions-footer">
                    {concealedVideoUrl && (
                      <button
                        type="button"
                        className="feature-download-btn"
                        onClick={downloadConcealedVideo}
                      >
                        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        <span>Download Shielded Video (.mp4)</span>
                      </button>
                    )}

                    <button
                      type="button"
                      className="mode-toggle-btn"
                      style={{ padding: '8px 16px', border: '1px solid rgba(255, 255, 255, 0.2)' }}
                      onClick={() => setWorkspaceTab('visualizer')}
                    >
                      Switch to Frame Visualizer
                    </button>
                  </div>
                </div>
              )}

            </div>
          )}

        </div>
      </div>
    </main>
  );
};
