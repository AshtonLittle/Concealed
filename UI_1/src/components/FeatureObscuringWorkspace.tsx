import React, { useState, useRef, useEffect } from 'react';

const COMMON_FEATURES = [
  'face',
  'person',
  'text',
  'bottle',
  'car',
  'chair',
  'dog',
  'laptop',
];

export const FeatureObscuringWorkspace: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [feature, setFeature] = useState('face');
  const [conf, setConf] = useState(0.25);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showContours, setShowContours] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [isProcessing, setIsProcessing] = useState(false);
  const [obscuredImageUrl, setObscuredImageUrl] = useState<string | null>(null);
  const [activeDisplayMode, setActiveDisplayMode] = useState<'obscured' | 'original'>('obscured');
  const [telemetry, setTelemetry] = useState<{
    feature: string;
    regionsCount: number;
    processingTimeMs: number;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Update object URL when file changes
  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      setObscuredImageUrl(null);
      setTelemetry(null);
      setErrorMsg(null);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
      setObscuredImageUrl(null);
      setTelemetry(null);
    }
  }, [selectedFile]);

  // Clean up obscured image object URL on unmount
  useEffect(() => {
    return () => {
      if (obscuredImageUrl) {
        URL.revokeObjectURL(obscuredImageUrl);
      }
    };
  }, [obscuredImageUrl]);

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

  const clearFile = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setObscuredImageUrl(null);
    setTelemetry(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleObscure = async () => {
    if (!selectedFile || !feature.trim()) return;

    setIsProcessing(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('feature', feature.trim());
    formData.append('conf', conf.toString());
    formData.append('show_boxes', showBoxes ? 'true' : 'false');
    formData.append('show_contours', showContours ? 'true' : 'false');
    formData.append('output_format', 'PNG');
    formData.append('response_type', 'image');

    try {
      const res = await fetch('http://127.0.0.1:8001/api/obscure-feature', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let detail = 'Feature obscuring failed.';
        try {
          const errJson = await res.json();
          detail = errJson.detail || detail;
        } catch {
          detail = `Server returned ${res.status}: ${res.statusText}`;
        }
        throw new Error(detail);
      }

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setObscuredImageUrl(objectUrl);
      setActiveDisplayMode('obscured');

      const regionsHeader = res.headers.get('X-Regions-Found') || '0';
      const timeHeader = res.headers.get('X-Processing-Time-Ms') || '0';
      setTelemetry({
        feature: feature.trim(),
        regionsCount: parseInt(regionsHeader, 10) || 0,
        processingTimeMs: parseFloat(timeHeader) || 0,
      });
    } catch (err: unknown) {
      console.error('Feature obscuring failed:', err);
      const msg = err instanceof Error ? err.message : 'Unknown connection error';
      setErrorMsg(
        `${msg}. Make sure the backend server is running via: python -m concealed.api.main`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!obscuredImageUrl) return;
    const link = document.createElement('a');
    link.href = obscuredImageUrl;
    link.download = `obscured_${feature.trim() || 'feature'}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="main-workspace feature-obscuring-workspace" aria-label="Feature Obscuring workspace">
      <div className="workspace-content feature-obscuring-content">
        
        {/* Workspace Title & Disclaimer Banner */}
      

        <div className="feature-obscuring-grid">
          {/* Left Column: Image Selection & Feature Controls */}
          <div className="feature-control-panel">
            
            {/* Step 1: Input Image Drop Box */}
            <div className="feature-panel-section">
              <label className="feature-section-label">1. SELECT IMAGE</label>
              <div
                className={`image-dropbox feature-image-dropbox ${isDragging ? 'is-dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                aria-label="Upload image for feature obscuring"
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
                      width="32"
                      height="32"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
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
                      title="Remove image"
                      aria-label="Remove image"
                    >
                      <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5">
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
                      width="36"
                      height="36"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                    <span className="dropbox-prompt-title">Drop an image here</span>
                    <span className="dropbox-prompt-subtitle">or click to browse</span>
                  </div>
                )}
              </div>
            </div>

            {/* Step 2: Feature Input */}
            <div className="feature-panel-section">
              <label className="feature-section-label" htmlFor="feature-input">
                2. TARGET FEATURE TO OBSCURE
              </label>
              <div className="feature-input-wrapper">
                <input
                  id="feature-input"
                  type="text"
                  className="feature-text-input"
                  placeholder="e.g. face, text, person, waterbottle, car..."
                  value={feature}
                  onChange={(e) => setFeature(e.target.value)}
                  disabled={isProcessing}
                />
              </div>

              {/* Quick Select Feature Chips */}
              <div className="feature-quick-chips">
                <span className="quick-chips-label">Quick select:</span>
                <div className="chips-container">
                  {COMMON_FEATURES.map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={`feature-chip ${feature.toLowerCase() === item ? 'active' : ''}`}
                      onClick={() => setFeature(item)}
                      disabled={isProcessing}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Step 3: Advanced Options */}
            <div className="feature-panel-section">
              <button
                type="button"
                className="feature-advanced-toggle"
                onClick={() => setShowAdvanced((prev) => !prev)}
              >
                <span>{showAdvanced ? '▼' : '►'} Advanced Detection Settings</span>
              </button>

              {showAdvanced && (
                <div className="feature-advanced-options">
                  <div className="advanced-option-row">
                    <label htmlFor="conf-slider" className="advanced-label">
                      Confidence Threshold: <strong>{(conf * 100).toFixed(0)}%</strong>
                    </label>
                    <input
                      id="conf-slider"
                      type="range"
                      min="0.05"
                      max="0.80"
                      step="0.05"
                      value={conf}
                      onChange={(e) => setConf(parseFloat(e.target.value))}
                      className="advanced-slider"
                    />
                  </div>

                  <div className="advanced-option-checkboxes">
                    <label className="advanced-checkbox-label">
                      <input
                        type="checkbox"
                        checked={showContours}
                        onChange={(e) => setShowContours(e.target.checked)}
                      />
                      <span>Highlight Conforming Contours</span>
                    </label>

                    <label className="advanced-checkbox-label">
                      <input
                        type="checkbox"
                        checked={showBoxes}
                        onChange={(e) => setShowBoxes(e.target.checked)}
                      />
                      <span>Show Bounding Boxes</span>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Error Message Alert */}
            {errorMsg && (
              <div className="feature-error-box" role="alert">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Action Button: OBSCURE FEATURE */}
            <button
              type="button"
              className="feature-action-btn"
              onClick={handleObscure}
              disabled={!selectedFile || !feature.trim() || isProcessing}
            >
              {isProcessing ? (
                <span className="feature-btn-loading">
                  <span className="spinner-dots" />
                  <span>OBSCURING '{feature}'...</span>
                </span>
              ) : (
                <span>OBSCURE FEATURE</span>
              )}
            </button>
          </div>

          {/* Right Column: Visual Preview & Results */}
          <div className="feature-preview-panel">
            <div className="feature-preview-card">
              <div className="preview-card-header">
                <span className="preview-card-title">VISUAL RESULT</span>
                {obscuredImageUrl && previewUrl && (
                  <div className="preview-mode-toggle" role="group" aria-label="Image comparison mode">
                    <button
                      type="button"
                      className={`mode-toggle-btn ${activeDisplayMode === 'original' ? 'active' : ''}`}
                      onClick={() => setActiveDisplayMode('original')}
                    >
                      Show Original
                    </button>
                    <button
                      type="button"
                      className={`mode-toggle-btn ${activeDisplayMode === 'obscured' ? 'active' : ''}`}
                      onClick={() => setActiveDisplayMode('obscured')}
                    >
                      Show Obscured
                    </button>
                  </div>
                )}
              </div>

              {/* Image Viewport */}
              <div className="preview-viewport">
                {isProcessing ? (
                  <div className="preview-processing-state">
                    <div className="preview-spinner" />
                    <span className="processing-headline">Running Silhouette Processor</span>
                    <span className="processing-subtext">
                      Grounding '{feature}' via YOLO & FastSAM conforming mask...
                    </span>
                  </div>
                ) : obscuredImageUrl ? (
                  <img
                    src={activeDisplayMode === 'obscured' ? obscuredImageUrl : (previewUrl || '')}
                    alt={activeDisplayMode === 'obscured' ? 'Obscured result' : 'Original image'}
                    className="preview-result-img"
                  />
                ) : previewUrl ? (
                  <div className="preview-placeholder-loaded">
                    <img src={previewUrl} alt="Original upload preview" className="preview-result-img muted" />
                    <div className="preview-overlay-hint">
                      Ready to obscure. Click <strong>OBSCURE FEATURE</strong> on the left.
                    </div>
                  </div>
                ) : (
                  <div className="preview-empty-state">
                    <svg
                      viewBox="0 0 24 24"
                      width="48"
                      height="48"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      className="preview-empty-icon"
                    >
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                    <span className="empty-state-title">No image selected</span>
                    <span className="empty-state-subtitle">Select an image to preview results here</span>
                  </div>
                )}
              </div>

              {/* Telemetry & Download footer */}
              {telemetry && obscuredImageUrl && (
                <div className="preview-telemetry-footer">
                  <div className="telemetry-badges-row">
                    <div className="telemetry-badge">
                      <span className="badge-k">TARGET</span>
                      <span className="badge-v highlight">{telemetry.feature}</span>
                    </div>
                    <div className="telemetry-badge">
                      <span className="badge-k">REGIONS</span>
                      <span className="badge-v">
                        {telemetry.regionsCount > 0 ? `${telemetry.regionsCount} found` : 'None found'}
                      </span>
                    </div>
                    <div className="telemetry-badge">
                      <span className="badge-k">LATENCY</span>
                      <span className="badge-v">{telemetry.processingTimeMs.toFixed(0)} ms</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="feature-download-btn"
                    onClick={handleDownload}
                  >
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    <span>Download Obscured Image</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

      </div>
    </main>
  );
};
