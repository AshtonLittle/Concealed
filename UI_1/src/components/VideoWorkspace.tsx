import React, { useState, useRef } from 'react';

export const VideoWorkspace: React.FC = () => {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const videoInputRef = useRef<HTMLInputElement>(null);

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
      if (file.type.startsWith('video/')) {
        setSelectedVideo(file);
        setVideoPreviewUrl(URL.createObjectURL(file));
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedVideo(file);
      setVideoPreviewUrl(URL.createObjectURL(file));
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
    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  return (
    <main className="main-workspace" aria-label="Video protection workspace">
      {/* Centered Video Protection Content */}
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

          {/* Headline & Tagline under video drop box */}
          <h2 className="protection-headline">
            Stay Concealed
          </h2>
          <p className="protection-tagline">
            Digital Camouflage
          </p>
        </div>
      </div>
    </main>
  );
};
