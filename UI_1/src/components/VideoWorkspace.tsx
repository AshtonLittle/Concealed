import React, { useState, useRef, useEffect } from 'react';
import type { ConcealStats } from './Sidebar';

interface VideoWorkspaceProps {
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const VideoWorkspace: React.FC<VideoWorkspaceProps> = ({ onStatsUpdate }) => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [concealedVideoUrl, setConcealedVideoUrl] = useState<string | null>(null);
  const [activeDisplayMode, setActiveDisplayMode] = useState<'concealed' | 'original'>('concealed');
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [telemetry, setTelemetry] = useState<{
    frames: number;
    fps: number;
    timeMs: number;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const videoInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      if (concealedVideoUrl) URL.revokeObjectURL(concealedVideoUrl);
    };
  }, [videoPreviewUrl, concealedVideoUrl]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/') || file.name.match(/\.(mp4|webm|mov|avi|mkv)$/i)) {
        setSelectedVideo(file);
        setVideoPreviewUrl(URL.createObjectURL(file));
        setConcealedVideoUrl(null);
        setTelemetry(null);
        setErrorMsg(null);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedVideo(file);
      setVideoPreviewUrl(URL.createObjectURL(file));
      setConcealedVideoUrl(null);
      setTelemetry(null);
      setErrorMsg(null);
    }
  };

  const triggerVideoInput = () => {
    videoInputRef.current?.click();
  };

  const clearVideo = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedVideo(null);
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
      setVideoPreviewUrl(null);
    }
    if (concealedVideoUrl) {
      URL.revokeObjectURL(concealedVideoUrl);
      setConcealedVideoUrl(null);
    }
    setTelemetry(null);
    setErrorMsg(null);
    onStatsUpdate?.(null);
    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  const handleConcealVideo = async () => {
    if (!selectedVideo) return;

    setIsProcessing(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedVideo);
    formData.append('epsilon', '8.0');
    formData.append('mode', 'hybrid');

    try {
      const res = await fetch('http://127.0.0.1:8001/api/obfuscate/video', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let detail = 'Video concealing failed.';
        try {
          const errJson = await res.json();
          detail = errJson.detail || detail;
        } catch {
          detail = `Server error ${res.status}: ${res.statusText}`;
        }
        throw new Error(detail);
      }

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setConcealedVideoUrl(objectUrl);
      setActiveDisplayMode('concealed');

      const frames = parseInt(res.headers.get('X-Frames-Processed') || '0', 10);
      const fps = parseFloat(res.headers.get('X-FPS') || '24');
      const timeMs = parseFloat(res.headers.get('X-Processing-Time-Ms') || '0');

      setTelemetry({
        frames: frames > 0 ? frames : 1,
        fps: fps > 0 ? fps : 24,
        timeMs: timeMs > 0 ? timeMs : 0,
      });

      onStatsUpdate?.({
        latencyMs: timeMs > 0 ? timeMs : null,
        qualityLossPct: null,
      });
    } catch (err: unknown) {
      console.error('Video concealing failed:', err);
      const msg = err instanceof Error ? err.message : 'Connection failed';
      setErrorMsg(`${msg}. Ensure backend is running on http://127.0.0.1:8001.`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadVideo = () => {
    if (!concealedVideoUrl) return;
    const link = document.createElement('a');
    link.href = concealedVideoUrl;
    link.download = `concealed_${selectedVideo?.name || 'video.mp4'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="main-workspace" aria-label="Video protection workspace">
      <div className="workspace-content">
        <div className="protection-status-center" aria-live="polite">
          
          {/* Video Drop Box */}
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
            aria-label="Select a video file"
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
                    <line x1="17" y1="17" x2="22" y2="17" />
                    <line x1="17" y1="7" x2="22" y2="7" />
                  </svg>
                )}
                <div className="dropbox-file-details">
                  <span className="dropbox-filename">{selectedVideo.name}</span>
                  <span className="dropbox-filesize">
                    {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB • Temporal Masking Ready
                  </span>
                </div>
                <button
                  type="button"
                  className="dropbox-clear-btn"
                  onClick={clearVideo}
                  title="Remove selected video"
                  aria-label="Remove selected video"
                >
                  <svg
                    viewBox="0 0 24 24"
                    width="12"
                    height="12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
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
                  width="42"
                  height="42"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.7"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <polygon points="23 7 16 12 23 17 23 7" />
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                </svg>
                <span className="dropbox-prompt-title">Select a video file</span>
                <span className="dropbox-prompt-subtitle">or drag and drop here</span>
              </div>
            )}
          </div>

          {/* Error Alert Box */}
          {errorMsg && (
            <div className="feature-error-box" style={{ maxWidth: '450px', marginBottom: '16px' }} role="alert">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Primary Action Button: CONCEAL VIDEO (Permanently Visible Pushbutton) */}
          {!concealedVideoUrl && (
            <div className="workspace-action-row">
              <button
                type="button"
                className="main-action-pushbutton"
                onClick={selectedVideo ? handleConcealVideo : triggerVideoInput}
                disabled={isProcessing}
                aria-label={selectedVideo ? "Conceal video and send to backend" : "Select video to conceal"}
              >
                {isProcessing ? (
                  <span className="action-btn-loading">
                    <span className="spinner-dots" />
                    <span>SENDING VIDEO TO BACKEND...</span>
                  </span>
                ) : (
                  <span className="pushbutton-content">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polygon points="23 7 16 12 23 17 23 7" />
                      <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                    </svg>
                    <span>{selectedVideo ? 'CONCEAL VIDEO (SEND TO BACKEND)' : 'SELECT VIDEO & CONCEAL'}</span>
                  </span>
                )}
              </button>
            </div>
          )}

          {/* Results Viewport & Controls */}
          {concealedVideoUrl && (
            <div style={{ width: '100%', maxWidth: '640px', marginBottom: '24px' }}>
              {/* Comparison Mode Toggle */}
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '14px' }}>
                <div className="preview-mode-toggle" role="group" aria-label="Comparison mode">
                  <button
                    type="button"
                    className={`mode-toggle-btn ${activeDisplayMode === 'original' ? 'active' : ''}`}
                    onClick={() => setActiveDisplayMode('original')}
                  >
                    Show Original
                  </button>
                  <button
                    type="button"
                    className={`mode-toggle-btn ${activeDisplayMode === 'concealed' ? 'active' : ''}`}
                    onClick={() => setActiveDisplayMode('concealed')}
                  >
                    Show Concealed Video
                  </button>
                </div>
              </div>

              {/* Video Player Display */}
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
                <video
                  src={activeDisplayMode === 'concealed' ? concealedVideoUrl : (videoPreviewUrl || '')}
                  controls
                  style={{
                    maxWidth: '100%',
                    maxHeight: '400px',
                    borderRadius: '10px',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    boxShadow: '0 8px 30px rgba(0, 0, 0, 0.6)',
                  }}
                />
              </div>

              {/* Telemetry Row */}
              {telemetry && (
                <div className="telemetry-badges-row" style={{ justifyContent: 'center', marginBottom: '16px' }}>
                  <div className="telemetry-badge">
                    <span className="badge-k">FRAMES</span>
                    <span className="badge-v highlight">{telemetry.frames}</span>
                  </div>
                  <div className="telemetry-badge">
                    <span className="badge-k">FPS</span>
                    <span className="badge-v">{telemetry.fps}</span>
                  </div>
                  <div className="telemetry-badge">
                    <span className="badge-k">LATENCY</span>
                    <span className="badge-v">{telemetry.timeMs.toFixed(0)} ms</span>
                  </div>
                  <div className="telemetry-badge">
                    <span className="badge-k">TEMPORAL SHIELD</span>
                    <span className="badge-v highlight">ACTIVE</span>
                  </div>
                </div>
              )}

              {/* Download & Re-process buttons */}
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button
                  type="button"
                  className="feature-download-btn"
                  onClick={handleDownloadVideo}
                >
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  <span>Download Concealed Video</span>
                </button>
                <button
                  type="button"
                  className="mode-toggle-btn"
                  style={{ padding: '8px 14px', border: '1px solid rgba(255, 255, 255, 0.2)' }}
                  onClick={handleConcealVideo}
                  disabled={isProcessing}
                >
                  Re-Conceal
                </button>
              </div>
            </div>
          )}

          {/* Headline & Tagline under drop box (only when no result yet) */}
          {!concealedVideoUrl && (
            <>
              <h2 className="protection-headline">
                Stay Concealed
              </h2>
              <p className="protection-tagline">
                Digital Camouflage
              </p>
            </>
          )}

        </div>
      </div>
    </main>
  );
};
