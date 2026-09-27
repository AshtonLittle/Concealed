import React, { useState, useRef } from 'react';
import type { SiglipOptionScore, SiglipProbeResponse } from '../types/probe';

interface ProbeWorkspaceProps {
  selectedModelEngine?: 'onnx' | 'pt';
  budget?: number;
  mode?: string;
}

const DEFAULT_OPTIONS = ['face', 'person', 'readable text', 'dog', 'car'];

export const ProbeWorkspace: React.FC<ProbeWorkspaceProps> = ({
  selectedModelEngine = 'onnx',
  budget = 8,
  mode = 'HYBRID',
}) => {
  // Image state
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string>('');
  const [imageSizeStr, setImageSizeStr] = useState<string>('');

  // Options state
  const [options, setOptions] = useState<string[]>(DEFAULT_OPTIONS);
  const [newOptionInput, setNewOptionInput] = useState<string>('');

  // Evaluation state
  const [isProbing, setIsProbing] = useState<boolean>(false);
  const [probeResult, setProbeResult] = useState<SiglipProbeResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // File handling
  const handleSelectFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please select a valid image file (JPEG, PNG, WEBP).');
      return;
    }
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeStr = file.size > 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : `${sizeKB} KB`;

    const url = URL.createObjectURL(file);
    setImageFile(file);
    setImagePreviewUrl(url);
    setImageName(file.name);
    setImageSizeStr(sizeStr);
    setErrorMsg(null);
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch('/test_image.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'test_image.jpg', { type: 'image/jpeg' });
      handleSelectFile(file);
    } catch {
      setErrorMsg('Could not load test image.');
    }
  };

  // Option management
  const handleAddOption = () => {
    const trimmed = newOptionInput.trim().toLowerCase();
    if (!trimmed) return;
    if (options.includes(trimmed)) {
      setNewOptionInput('');
      return;
    }
    setOptions((prev) => [...prev, trimmed]);
    setNewOptionInput('');
  };

  const handleKeyDownAdd = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddOption();
    }
  };

  const handleRemoveOption = (optToRemove: string) => {
    setOptions((prev) => prev.filter((o) => o !== optToRemove));
  };

  const handleAddPreset = (presetList: string[]) => {
    setOptions((prev) => {
      const merged = [...prev];
      presetList.forEach((p) => {
        const lower = p.toLowerCase();
        if (!merged.includes(lower)) merged.push(lower);
      });
      return merged;
    });
  };

  const handleClearOptions = () => {
    setOptions([]);
  };

  // Run Real SigLIP probe
  const handleRunSiglipProbe = async () => {
    if (!imageFile) {
      setErrorMsg('Please select or upload an image to probe.');
      return;
    }
    if (options.length === 0) {
      setErrorMsg('Please provide at least one option to test.');
      return;
    }

    setIsProbing(true);
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('options', options.join(', '));
      formData.append('model_engine', selectedModelEngine);
      formData.append('epsilon', String(budget));
      formData.append('mode', mode);

      const res = await fetch('http://127.0.0.1:8001/api/probe/siglip', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let detail = `Server HTTP error ${res.status}`;
        try {
          const errData = await res.json();
          detail = errData.detail || detail;
        } catch {}
        throw new Error(detail);
      }

      const data: SiglipProbeResponse = await res.json();
      setProbeResult(data);
    } catch (err: unknown) {
      console.error('SigLIP probe error:', err);
      const msg = err instanceof Error ? err.message : 'Evaluation failed';
      setErrorMsg(`${msg}. Verify Concealed backend is running on http://127.0.0.1:8001.`);
    } finally {
      setIsProbing(false);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDragLeave = () => {
    setIsDraggingOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="siglip-probe-workspace">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleSelectFile(e.target.files[0]);
          }
        }}
      />

      {/* TOP HEADER */}
      <div className="siglip-header-bar">
        <div className="siglip-header-title-group">
          <h1 className="siglip-workspace-title">SigLIP Feature Probe</h1>
          <p className="siglip-workspace-sub">
            Test what Google SigLIP detects before and after Concealed protection for your options.
          </p>
        </div>
        <div className="siglip-header-tags-group">
          <div className="siglip-engine-tag">
            <span className="siglip-model-label">Probe:</span>
            <strong>google/siglip-base-patch16-224</strong>
          </div>
          <div className="siglip-engine-tag params-tag">
            <span className="siglip-model-label">Concealing Params:</span>
            <strong>
              {selectedModelEngine === 'onnx' ? 'ONNX' : 'PyTorch'} • ε={budget} • {mode}
            </strong>
          </div>
        </div>
      </div>

      {/* ERROR TOAST */}
      {errorMsg && (
        <div className="siglip-error-banner" role="alert">
          <span>{errorMsg}</span>
          <button type="button" className="error-close-btn" onClick={() => setErrorMsg(null)}>
            ✕
          </button>
        </div>
      )}

      {/* MAIN TWO-COLUMN CONFIGURATION ROW */}
      <div className="siglip-controls-grid">
        {/* LEFT COLUMN: IMAGE UPLOAD & PREVIEW */}
        <div className="siglip-card image-card">
          <div className="siglip-card-header">
            <span className="siglip-card-title">Image</span>
            {imageFile && (
              <button
                type="button"
                className="siglip-action-btn secondary"
                onClick={() => fileInputRef.current?.click()}
              >
                Change Image
              </button>
            )}
          </div>

          {!imagePreviewUrl ? (
            <div
              className={`siglip-dropzone ${isDraggingOver ? 'dragging' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dropzone-icon">
                <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="dropzone-title">Click to upload or drag & drop image</p>
              <p className="dropzone-sub">Supports PNG, JPEG, WEBP</p>
              <button
                type="button"
                className="siglip-sample-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  handleLoadSample();
                }}
              >
                Use Test Image
              </button>
            </div>
          ) : (
            <div className="siglip-image-previews-container">
              <div className="preview-split">
                <div className="preview-pane">
                  <span className="preview-label">Original Image</span>
                  <img src={imagePreviewUrl} alt="Original" className="preview-img" />
                </div>
                <div className="preview-pane">
                  <span className="preview-label">Protected Image</span>
                  <img
                    src={probeResult?.concealed_image_url || imagePreviewUrl}
                    alt="Protected"
                    className={`preview-img ${!probeResult ? 'pending' : ''}`}
                  />
                  {!probeResult && <span className="preview-overlay-note">Run probe to view</span>}
                </div>
              </div>
              <div className="image-meta-bar">
                <span className="meta-name">{imageName}</span>
                <span className="meta-size">{imageSizeStr}</span>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: OPTIONS MANAGEMENT */}
        <div className="siglip-card options-card">
          <div className="siglip-card-header">
            <span className="siglip-card-title">Options to Test</span>
            <span className="siglip-count-badge">{options.length} options</span>
          </div>

          <p className="options-hint">
            Provide the features, concepts, or labels for SigLIP to evaluate. No text bar needed.
          </p>

          {/* Quick Preset Buttons */}
          <div className="options-preset-bar">
            <span className="preset-label">Add Presets:</span>
            <button
              type="button"
              className="preset-chip"
              onClick={() => handleAddPreset(['face', 'person', 'identity', 'smile'])}
            >
              + Face & Identity
            </button>
            <button
              type="button"
              className="preset-chip"
              onClick={() => handleAddPreset(['text', 'document', 'license plate', 'numbers'])}
            >
              + Text & Numbers
            </button>
            <button
              type="button"
              className="preset-chip"
              onClick={() => handleAddPreset(['car', 'vehicle', 'dog', 'building'])}
            >
              + Objects
            </button>
            {options.length > 0 && (
              <button type="button" className="preset-chip clear" onClick={handleClearOptions}>
                Clear All
              </button>
            )}
          </div>

          {/* Active Option Tags */}
          <div className="options-tags-list">
            {options.map((opt) => (
              <div key={opt} className="option-tag">
                <span className="option-tag-text">{opt}</span>
                <button
                  type="button"
                  className="option-tag-remove"
                  onClick={() => handleRemoveOption(opt)}
                  title={`Remove ${opt}`}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {/* Add Option Input Row */}
          <div className="add-option-row">
            <input
              type="text"
              className="add-option-input"
              placeholder="Type an option and press Enter..."
              value={newOptionInput}
              onChange={(e) => setNewOptionInput(e.target.value)}
              onKeyDown={handleKeyDownAdd}
            />
            <button
              type="button"
              className="add-option-btn"
              onClick={handleAddOption}
              disabled={!newOptionInput.trim()}
            >
              + Add Option
            </button>
          </div>

          {/* Action Button */}
          <button
            type="button"
            className="siglip-run-btn"
            disabled={!imageFile || options.length === 0 || isProbing}
            onClick={handleRunSiglipProbe}
          >
            {isProbing ? (
              <>
                <span className="btn-spinner" />
                Evaluating with SigLIP...
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Probe Options with SigLIP
              </>
            )}
          </button>
        </div>
      </div>

      {/* RESULTS SECTION */}
      {probeResult && (
        <div className="siglip-results-section">
          {/* Summary Stat Cards */}
          <div className="siglip-summary-banner">
            <div className="siglip-stat-card primary">
              <span className="stat-label">Privacy Protection</span>
              <span className="stat-val highlight">{probeResult.overall_protection_pct}%</span>
            </div>
            <div className="siglip-stat-card">
              <span className="stat-label">Options Hidden</span>
              <span className="stat-val">
                {probeResult.options_hidden_count} / {probeResult.total_options}
              </span>
            </div>
            <div className="siglip-stat-card">
              <span className="stat-label">Avg. Confidence Lost</span>
              <span className="stat-val drop">-{probeResult.avg_confidence_drop_pct}%</span>
            </div>
          </div>

          {/* Scores Table */}
          <div className="siglip-scores-card">
            <div className="scores-card-header">
              <span className="scores-card-title">Real SigLIP Confidence Scores</span>
              <span className="scores-card-badge">Zero-Shot Vision Evaluation</span>
            </div>

            <div className="siglip-scores-table">
              <div className="siglip-table-header">
                <span className="col-option">Option</span>
                <span className="col-score">Original Image</span>
                <span className="col-score">Protected Image</span>
                <span className="col-drop">Confidence Lost</span>
                <span className="col-status">Result</span>
              </div>

              {probeResult.results.map((r: SiglipOptionScore, idx: number) => {
                const isHidden = r.status === 'Hidden';
                const isWeakened = r.status === 'Weakened';
                const pillClass = isHidden ? 'hidden' : isWeakened ? 'weakened' : 'visible';

                return (
                  <div key={idx} className="siglip-table-row">
                    <span className="col-option option-name">{r.option}</span>

                    <div className="col-score">
                      <span className="score-num clean">{r.clean_confidence_pct}%</span>
                      <div className="mini-track">
                        <div
                          className="mini-fill clean"
                          style={{ width: `${Math.min(100, Math.max(0, r.clean_confidence_pct))}%` }}
                        />
                      </div>
                    </div>

                    <div className="col-score">
                      <span className="score-num concealed">{r.concealed_confidence_pct}%</span>
                      <div className="mini-track">
                        <div
                          className="mini-fill concealed"
                          style={{ width: `${Math.min(100, Math.max(0, r.concealed_confidence_pct))}%` }}
                        />
                      </div>
                    </div>

                    <div className="col-drop">
                      <span className="drop-badge">-{r.confidence_drop_pct}%</span>
                    </div>

                    <div className="col-status">
                      <span className={`status-pill ${pillClass}`}>{r.status}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
