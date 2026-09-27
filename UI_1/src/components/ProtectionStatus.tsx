import React, { useState, useRef, useEffect } from 'react';
import type { ConcealStats } from './Sidebar';
import type { ProtectionMode, ImageOutputFormat } from './SettingsSidebar';

interface ProtectionStatusProps {
  selectedModelEngine?: 'onnx' | 'pt';
  budget?: number;
  mode?: ProtectionMode;
  outputFormat?: ImageOutputFormat;
  onStatsUpdate?: (stats: ConcealStats | null) => void;
}

export const ProtectionStatus: React.FC<ProtectionStatusProps> = ({
  selectedModelEngine = 'onnx',
  budget = 8,
  mode = 'HYBRID',
  outputFormat = 'PNG',
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
  const originalFileRef = useRef<File | null>(null);

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
      const file = e.dataTransfer.files[0];
      originalFileRef.current = file;
      setSelectedFile(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      originalFileRef.current = file;
      setSelectedFile(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    originalFileRef.current = null;
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
    const fileToConceal = originalFileRef.current || selectedFile;
    if (!fileToConceal) return;

    setIsProcessing(true);
    setErrorMsg(null);

    const modeParam = mode === 'RESIDUAL' ? 'canonical_residual' : mode.toLowerCase();

    const formData = new FormData();
    formData.append('file', fileToConceal);
    formData.append('epsilon', budget.toString());
    formData.append('mode', modeParam);
    formData.append('texture_masking', 'true');
    formData.append('chroma_damping', '0.7');
    formData.append('strip_metadata', 'true');
    formData.append('output_format', outputFormat);
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
      const linf = parseFloat(res.headers.get('X-Linf-255') || budget.toString());
      const timeMs = parseFloat(res.headers.get('X-Processing-Time-Ms') || '0');
      const rawLoss = parseFloat(res.headers.get('X-Quality-Loss-Pct') || '');
      const qualityLossPct = Number.isFinite(rawLoss)
        ? rawLoss
        : (Number.isFinite(ssim) && ssim > 0 ? Math.max(0, (1 - ssim) * 100) : 0);

      const model = res.headers.get('X-Model') || (selectedModelEngine === 'onnx' ? 'generator.onnx' : 'best_generator.pt');

      setTelemetry({
        model,
        psnr: Number.isFinite(psnr) ? psnr : 42.0,
        ssim: Number.isFinite(ssim) ? ssim : 0.98,
        linf: Number.isFinite(linf) ? linf : budget,
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
    const ext = outputFormat.toLowerCase();
    const actualExt = ext === 'jpeg' ? 'jpg' : ext;
    const baseName = selectedFile?.name ? selectedFile.name.replace(/\.[^/.]+$/, '') : 'protected';
    link.download = `concealed_${baseName}.${actualExt}`;
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
        <div className="workspace-action-row" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
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

          {/* Dynamic Settings Status Line */}
          <div className="workspace-settings-summary-strip" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: '#94a3b8', letterSpacing: '0.04em' }}>
            <span>Engine: <strong style={{ color: '#e2e8f0' }}>{selectedModelEngine === 'onnx' ? 'ONNX Runtime' : 'PyTorch'}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Budget: <strong style={{ color: '#e2e8f0' }}>ε = {budget}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Mode: <strong style={{ color: '#e2e8f0' }}>{mode}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Format: <strong style={{ color: '#e2e8f0' }}>{outputFormat}</strong></span>
          </div>
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

          {/* Image Display with In-Flight Re-Conceal Loading Overlay */}
          <div style={{ position: 'relative', display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
            <img
              src={activeDisplayMode === 'concealed' ? concealedImageUrl : (previewUrl || '')}
              alt={activeDisplayMode === 'concealed' ? 'Concealed result' : 'Original input'}
              style={{
                maxWidth: '100%',
                maxHeight: '400px',
                borderRadius: '10px',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                boxShadow: '0 8px 30px rgba(0, 0, 0, 0.6)',
                opacity: isProcessing ? 0.45 : 1,
                transition: 'opacity 0.2s ease',
              }}
            />
            {isProcessing && (
              <div
                style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(15, 15, 18, 0.90)',
                  padding: '12px 22px',
                  borderRadius: '10px',
                  border: '1px solid rgba(168, 85, 247, 0.45)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  color: '#f1f1f5',
                  fontSize: '12.5px',
                  fontWeight: 600,
                  boxShadow: '0 8px 28px rgba(0, 0, 0, 0.7)',
                  pointerEvents: 'none',
                }}
              >
                <span className="spinner-dots" />
                <span>Re-concealing original image with active settings...</span>
              </div>
            )}
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
                <span className="badge-k">MODE</span>
                <span className="badge-v">{mode}</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">BUDGET</span>
                <span className="badge-v">ε = {budget}</span>
              </div>
              <div className="telemetry-badge">
                <span className="badge-k">FORMAT</span>
                <span className="badge-v">{outputFormat}</span>
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

          {/* Active Target Settings & Source Strip */}
          <div
            className="workspace-settings-summary-strip"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexWrap: 'wrap',
              gap: '8px',
              fontSize: '11px',
              color: '#94a3b8',
              letterSpacing: '0.04em',
              marginBottom: '14px',
            }}
          >
            <span>Target: <strong style={{ color: '#e2e8f0' }}>{selectedFile?.name || 'Input Image'} (Original)</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Engine: <strong style={{ color: '#e2e8f0' }}>{selectedModelEngine === 'onnx' ? 'ONNX Runtime' : 'PyTorch'}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Budget: <strong style={{ color: '#e2e8f0' }}>ε = {budget}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Mode: <strong style={{ color: '#e2e8f0' }}>{mode}</strong></span>
            <span style={{ opacity: 0.3 }}>•</span>
            <span>Format: <strong style={{ color: '#e2e8f0' }}>{outputFormat}</strong></span>
          </div>

          {/* Download & Re-process buttons */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', alignItems: 'center' }}>
            <button
              type="button"
              className="feature-download-btn"
              onClick={handleDownload}
              title="Download currently protected image"
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
              className="feature-reconceal-btn"
              onClick={handleConceal}
              disabled={isProcessing}
              title={`Re-run protection pipeline on the original uploaded image (${selectedFile?.name || 'original'}) with current settings`}
            >
              {isProcessing ? (
                <>
                  <span className="spinner-dots" />
                  <span>Re-Concealing Original...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <polyline points="23 4 23 10 17 10" />
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                  </svg>
                  <span>Re-Conceal Original Image</span>
                </>
              )}
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
