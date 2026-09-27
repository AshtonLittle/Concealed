import React, { useState, useRef } from 'react';

export const ProtectionStatus: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="protection-status-center" aria-live="polite">
      {/* Drop Box for Image / File Selection */}
      <div
        className={`image-dropbox ${isDragging ? 'is-dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerFileInput}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            triggerFileInput();
          }
        }}
        aria-label="Select an image or file"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="dropbox-hidden-input"
          aria-hidden="true"
        />

        {selectedFile ? (
          <div className="dropbox-file-info">
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
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            <div className="dropbox-file-details">
              <span className="dropbox-filename">{selectedFile.name}</span>
              <span className="dropbox-filesize">
                {(selectedFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
            <button
              type="button"
              className="dropbox-clear-btn"
              onClick={clearFile}
              title="Remove selected file"
              aria-label="Remove selected file"
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="dropbox-content">
            <svg
              className="dropbox-upload-icon"
              viewBox="0 0 24 24"
              width="40"
              height="40"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span className="dropbox-prompt-title">Select an image or file</span>
            <span className="dropbox-prompt-subtitle">or drag and drop here</span>
          </div>
        )}
      </div>

      {/* Slogan under drop box */}
      <h2 className="protection-headline">
        Stay Safe. Not Sorry
      </h2>

      {/* Subtitle / Status Line */}
      <div className="protection-status-line">
        <span className="status-indicator-dot" aria-hidden="true" />
        <span className="protection-submessage">
          IMAGE PROTECTION ACTIVE
        </span>
      </div>
    </div>
  );
};
