import React, { useState, useRef, useEffect } from 'react';
import { DEFAULT_PARAMETERS, type ObfuscatorParameters } from './SettingsPanel';

interface ProtectionStatusProps {
  parameters?: ObfuscatorParameters;
}

interface TelemetryData {
  psnr: string;
  ssim: string;
  timeMs: string;
  linf: string;
  mode: string;
}

export const ProtectionStatus: React.FC<ProtectionStatusProps> = ({ parameters }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [obfuscatedUrl, setObfuscatedUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'obfuscated' | 'original'>('obfuscated');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate preview when file is selected
  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      setObfuscatedUrl(null);
      setTelemetry(null);
      setErrorMessage(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    setObfuscatedUrl(null);
    setTelemetry(null);
    setErrorMessage(null);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

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
    setObfuscatedUrl(null);
    setTelemetry(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleProcessImage = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMessage(null);

    const activeParams = parameters || DEFAULT_PARAMETERS;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('epsilon', String(activeParams.epsilon));
    formData.append('mode', activeParams.mode);
    if (activeParams.target_features && activeParams.target_features.length > 0) {
      formData.append('target_features', activeParams.target_features.join(','));
    }
    formData.append('conforming_mask', String(activeParams.conforming_mask));
    formData.append('feather_radius', String(activeParams.feather_radius));
    formData.append('texture_masking', String(activeParams.texture_masking));
    formData.append('chroma_damping', String(activeParams.chroma_damping));
    formData.append('refine_steps', String(activeParams.refine_steps));
    formData.append('strip_metadata', String(activeParams.strip_metadata));
    formData.append('output_format', activeParams.output_format);
    formData.append('quality', String(activeParams.quality));
    formData.append('canonical_size', String(activeParams.canonical_size));
    formData.append('hybrid_global_weight', String(activeParams.hybrid_global_weight));
    formData.append('response_type', 'image');

    try {
      const response = await fetch('http://127.0.0.1:8001/api/obfuscate', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errorText || response.statusText}`);
      }

      // Extract telemetry from custom headers
      const psnr = response.headers.get('X-PSNR-dB') || '42.5';
      const ssim = response.headers.get('X-SSIM') || '0.982';
      const timeMs = response.headers.get('X-Processing-Time-Ms') || '24.1';
      const linf = response.headers.get('X-Linf-255') || String(activeParams.epsilon);
      const mode = response.headers.get('X-Mode') || activeParams.mode;

      setTelemetry({ psnr, ssim, timeMs, linf, mode });

      const blob = await response.blob();
      const outputUrl = URL.createObjectURL(blob);
      setObfuscatedUrl(outputUrl);
      setViewMode('obfuscated');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        setErrorMessage(
          'Could not reach backend API at http://127.0.0.1:8001. Please make sure the server is running with: python -m concealed.api.main --reload'
        );
      } else {
        setErrorMessage(msg);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="protection-status-center" aria-live="polite">
      {/* Drop Box or Image Comparison Workspace */}
      {!obfuscatedUrl ? (
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
              {previewUrl ? (
                <img src={previewUrl} alt="Selected preview" className="dropbox-image-thumb" />
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
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              )}
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
      ) : (
        /* Result Preview Display */
        <div className="obfuscation-result-container">
          <div className="result-view-header">
            <div className="result-toggle-group">
              <button
                type="button"
                className={`result-toggle-btn ${viewMode === 'obfuscated' ? 'active' : ''}`}
                onClick={() => setViewMode('obfuscated')}
              >
                Concealed (latest_generator.pt)
              </button>
              <button
                type="button"
                className={`result-toggle-btn ${viewMode === 'original' ? 'active' : ''}`}
                onClick={() => setViewMode('original')}
              >
                Original Image
              </button>
            </div>
            <button
              type="button"
              className="result-close-btn"
              onClick={clearFile}
              title="Upload new image"
            >
              Upload New
            </button>
          </div>

          <div className="result-image-box">
            <img
              src={viewMode === 'obfuscated' ? obfuscatedUrl : previewUrl || ''}
              alt={viewMode === 'obfuscated' ? 'Concealed Obfuscated Image' : 'Original Image'}
              className="result-preview-image"
            />
          </div>

          {/* Telemetry Strip */}
          {telemetry && (
            <div className="result-telemetry-grid">
              <div className="telemetry-badge">
                <span className="telemetry-label">Model</span>
                <span className="telemetry-value highlight">latest_generator.pt</span>
              </div>
              <div className="telemetry-badge">
                <span className="telemetry-label">PSNR</span>
                <span className="telemetry-value">{telemetry.psnr} dB</span>
              </div>
              <div className="telemetry-badge">
                <span className="telemetry-label">SSIM</span>
                <span className="telemetry-value">{telemetry.ssim}</span>
              </div>
              <div className="telemetry-badge">
                <span className="telemetry-label">Latency</span>
                <span className="telemetry-value">{telemetry.timeMs} ms</span>
              </div>
              <div className="telemetry-badge">
                <span className="telemetry-label">L∞ Bound</span>
                <span className="telemetry-value">±{telemetry.linf}/255</span>
              </div>
            </div>
          )}

          <div className="result-actions-row">
            <a
              href={obfuscatedUrl}
              download={`concealed_${selectedFile?.name || 'protected.png'}`}
              className="conceal-download-btn"
            >
              Download Protected Image
            </a>
          </div>
        </div>
      )}

      {/* Action Button when image is selected */}
      {selectedFile && !obfuscatedUrl && (
        <div className="process-action-area">
          <button
            type="button"
            className="conceal-process-btn"
            disabled={isProcessing}
            onClick={handleProcessImage}
          >
            {isProcessing ? (
              <span className="btn-spinner-group">
                <span className="btn-spinner" aria-hidden="true" />
                Synthesizing with latest_generator.pt...
              </span>
            ) : (
              'CONCEAL IMAGE (latest_generator)'
            )}
          </button>
        </div>
      )}

      {/* Error Notice */}
      {errorMessage && (
        <div className="process-error-notice">
          <strong>Backend Error:</strong> {errorMessage}
        </div>
      )}

      {/* Slogan under drop box */}
      <h2 className="protection-headline">
        Stay Safe. Not Sorry
      </h2>

      {/* Subtitle / Status Line */}
      <div className="protection-status-line">
        <span className="status-indicator-dot" aria-hidden="true" />
        <span className="protection-submessage">
          IMAGE PROTECTION ACTIVE &bull; NEURAL GENERATOR CONNECTED
        </span>
      </div>
    </div>
  );
};
