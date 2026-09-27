import React, { useState, useRef, useEffect } from 'react';
import type { ConcealStats } from './Sidebar';

interface ProtectionStatusProps {
  selectedModelEngine?: 'onnx' | 'pt';
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const ProtectionStatus: React.FC<ProtectionStatusProps> = ({
  selectedModelEngine = 'onnx',
  onStatsUpdate,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [concealedImageUrl, setConcealedImageUrl] = useState<string | null>(null);
  const [activeDisplayMode, setActiveDisplayMode] = useState<'concealed' | 'original'>('concealed');
  const [telemetry, setTelemetry] = useState<{
    model: string;
    psnr: number;
    ssim: number;
    linf: number;
    timeMs: number;
  } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      setConcealedImageUrl(null);
      setTelemetry(null);
      setErrorMsg(null);
      onStatsUpdate?.(null);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
      setConcealedImageUrl(null);
      setTelemetry(null);
      onStatsUpdate?.(null);
    }
  }, [selectedFile, onStatsUpdate]);

  useEffect(() => {
    return () => {
      if (concealedImageUrl) {
        URL.revokeObjectURL(concealedImageUrl);
      }
    };
  }, [concealedImageUrl]);

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
    setPreviewUrl(null);
    setConcealedImageUrl(null);
    setTelemetry(null);
    setErrorMsg(null);
    onStatsUpdate?.(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleConceal = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('epsilon', '8.0');
    formData.append('mode', 'hybrid');
    formData.append('texture_masking', 'true');
    formData.append('chroma_damping', '0.7');
    formData.append('strip_metadata', 'true');
    formData.append('output_format', 'ORIGINAL');
    formData.append('response_type', 'image');
    formData.append('model_engine', selectedModelEngine);

    try {
      const res = await fetch('http://127.0.0.1:8001/api/obfuscate', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let detail = 'Image concealing failed.';
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
      setConcealedImageUrl(objectUrl);
      setActiveDisplayMode('concealed');

      const psnr = parseFloat(res.headers.get('X-PSNR-dB') || '0');
      const ssim = parseFloat(res.headers.get('X-SSIM') || '0');
      const linf = parseFloat(res.headers.get('X-Linf-255') || '8.0');
      const timeMs = parseFloat(res.headers.get('X-Processing-Time-Ms') || '0');
      const rawLoss = parseFloat(res.headers.get('X-Quality-Loss-Pct') || '');
      const qualityLossPct = Number.isFinite(rawLoss)
        ? rawLoss
        : (Number.isFinite(ssim) && ssim > 0 ? Math.max(0, (1 - ssim) * 100) : 0);

      const model = res.headers.get('X-Model') || 'best_generator.pt';

      setTelemetry({
        model,
        psnr: Number.isFinite(psnr) ? psnr : 42.0,
        ssim: Number.isFinite(ssim) ? ssim : 0.98,
        linf: Number.isFinite(linf) ? linf : 8.0,
        timeMs: Number.isFinite(timeMs) ? timeMs : 0,
      });

      onStatsUpdate?.({
        latencyMs: Number.isFinite(timeMs) ? timeMs : null,
        qualityLossPct: Number.isFinite(qualityLossPct) ? qualityLossPct : null,
        psnrDb: Number.isFinite(psnr) ? psnr : null,
        ssim: Number.isFinite(ssim) ? ssim : null,
      });
    } catch (err: unknown) {
      console.error('Image concealing failed:', err);
      const msg = err instanceof Error ? err.message : 'Connection failed';
      setErrorMsg(`${msg}. Ensure backend is running via: python -m concealed.api.main`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!concealedImageUrl) return;
    const link = document.createElement('a');
    link.href = concealedImageUrl;
    link.download = `concealed_${selectedFile?.name || 'protected.png'}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="protection-status-center" aria-live="polite">
      {/* Drop Box for Image Selection */}
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
            <span className="dropbox-prompt-title">Select an image</span>
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

      {/* Primary Action Button: CONCEAL IMAGE (Permanently Visible Pushbutton) */}
      {!concealedImageUrl && (
        <div className="workspace-action-row">
          <button
            type="button"
            className="main-action-pushbutton"
            onClick={selectedFile ? handleConceal : triggerFileInput}
            disabled={isProcessing}
            aria-label={selectedFile ? "Conceal image and send to backend" : "Select image to conceal"}
          >
            {isProcessing ? (
              <span className="action-btn-loading">
                <span className="spinner-dots" />
                <span>SENDING TO BACKEND & CONCEALING...</span>
              </span>
            ) : (
              <span className="pushbutton-content">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
                <span>{selectedFile ? 'CONCEAL IMAGE (SEND TO BACKEND)' : 'SELECT IMAGE & CONCEAL'}</span>
              </span>
            )}
          </button>
        </div>
      )}

      {/* Results Viewport & Controls */}
      {concealedImageUrl && (
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
                Show Concealed
              </button>
            </div>
          </div>

          {/* Image Display */}
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
            <img
              src={activeDisplayMode === 'concealed' ? concealedImageUrl : (previewUrl || '')}
              alt={activeDisplayMode === 'concealed' ? 'Concealed result' : 'Original input'}
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
                <span className="badge-k">ENGINE</span>
                <span className="badge-v highlight">
                  {selectedModelEngine === 'onnx' ? 'generator.onnx' : (telemetry.model || 'best_generator.pt')}
                </span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">PSNR</span>
                <span className="badge-v">{telemetry.psnr.toFixed(1)} dB</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">SSIM</span>
                <span className="badge-v">{telemetry.ssim.toFixed(4)}</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">QUALITY LOSS</span>
                <span className="badge-v">{((1.0 - telemetry.ssim) * 100).toFixed(1)}%</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">L_INF</span>
                <span className="badge-v">{telemetry.linf.toFixed(1)}/255</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">LATENCY</span>
                <span className="badge-v">{telemetry.timeMs.toFixed(0)} ms</span>
              </div>
            </div>
          )}

          {/* Download & Re-process buttons */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
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
              <span>Download Protected Image</span>
            </button>
            <button
              type="button"
              className="mode-toggle-btn"
              style={{ padding: '8px 14px', border: '1px solid rgba(255, 255, 255, 0.2)' }}
              onClick={handleConceal}
              disabled={isProcessing}
            >
              Re-Conceal
            </button>
          </div>
        </div>
      )}

      {/* Headline & Tagline under drop box (only when no result yet) */}
      {!concealedImageUrl && (
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
  );
};
