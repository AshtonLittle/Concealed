import React, { useState, useEffect, useRef } from 'react';
import { CloseIcon } from './Icons';

export interface ObfuscatorParameters {
  epsilon: number;
  mode: 'hybrid' | 'canonical_residual' | 'native';
  target_features: string[];
  conforming_mask: boolean;
  feather_radius: number;
  texture_masking: boolean;
  chroma_damping: number;
  refine_steps: number;
  strip_metadata: boolean;
  output_format: 'PNG' | 'JPEG' | 'WEBP' | 'ORIGINAL';
  quality: number;
  canonical_size: number;
  hybrid_global_weight: number;
  response_type: 'image' | 'json';
}

export const DEFAULT_PARAMETERS: ObfuscatorParameters = {
  epsilon: 8.0,
  mode: 'hybrid',
  target_features: [],
  conforming_mask: false,
  feather_radius: 8,
  texture_masking: true,
  chroma_damping: 0.7,
  refine_steps: 0,
  strip_metadata: true,
  output_format: 'PNG',
  quality: 95,
  canonical_size: 384,
  hybrid_global_weight: 0.6,
  response_type: 'image',
};

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  parameters?: ObfuscatorParameters;
  onChangeParameters?: (params: ObfuscatorParameters) => void;
}

type TabType = 'all' | 'evasion' | 'masking' | 'stealth' | 'output';

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  isOpen,
  onClose,
  parameters,
  onChangeParameters,
}) => {
  const [params, setParams] = useState<ObfuscatorParameters>(parameters || DEFAULT_PARAMETERS);
  const [activeTab, setActiveTab] = useState<TabType>('all');
  const modalRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Sync external parameters if provided
  useEffect(() => {
    if (parameters) {
      setParams(parameters);
    }
  }, [parameters]);

  const updateParam = <K extends keyof ObfuscatorParameters>(key: K, value: ObfuscatorParameters[K]) => {
    const next = { ...params, [key]: value };
    setParams(next);
    onChangeParameters?.(next);
  };

  const resetDefaults = () => {
    setParams(DEFAULT_PARAMETERS);
    onChangeParameters?.(DEFAULT_PARAMETERS);
  };

  // Keyboard accessibility
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    closeBtnRef.current?.focus();

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const toggleFeature = (feat: string) => {
    let next: string[];
    if (feat === 'all') {
      next = params.target_features.includes('all') ? [] : ['all'];
    } else {
      const withoutAll = params.target_features.filter((f) => f !== 'all');
      if (withoutAll.includes(feat)) {
        next = withoutAll.filter((f) => f !== feat);
      } else {
        next = [...withoutAll, feat];
      }
    }
    updateParam('target_features', next);
  };

  const showEvasion = activeTab === 'all' || activeTab === 'evasion';
  const showMasking = activeTab === 'all' || activeTab === 'masking';
  const showStealth = activeTab === 'all' || activeTab === 'stealth';
  const showOutput = activeTab === 'all' || activeTab === 'output';

  return (
    <div className="settings-overlay" onClick={onClose} role="presentation">
      <div
        className="settings-modal"
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="settings-header">
          <div className="settings-title-group">
            <h3 id="settings-title" className="settings-heading">
              CONCEALED PARAMETERS
            </h3>
            <span className="settings-subheading">
              AI Obfuscator Catalog v0.1.0
            </span>
          </div>
          <button
            ref={closeBtnRef}
            type="button"
            className="settings-close-btn"
            onClick={onClose}
            aria-label="Close settings dialog"
          >
            <CloseIcon size={14} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="settings-tab-bar" role="tablist" aria-label="Settings Categories">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'all'}
            className={`settings-tab-btn ${activeTab === 'all' ? 'active' : ''}`}
            onClick={() => setActiveTab('all')}
          >
            All
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'evasion'}
            className={`settings-tab-btn ${activeTab === 'evasion' ? 'active' : ''}`}
            onClick={() => setActiveTab('evasion')}
          >
            AI Evasion
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'masking'}
            className={`settings-tab-btn ${activeTab === 'masking' ? 'active' : ''}`}
            onClick={() => setActiveTab('masking')}
          >
            Masking
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'stealth'}
            className={`settings-tab-btn ${activeTab === 'stealth' ? 'active' : ''}`}
            onClick={() => setActiveTab('stealth')}
          >
            Stealth
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'output'}
            className={`settings-tab-btn ${activeTab === 'output' ? 'active' : ''}`}
            onClick={() => setActiveTab('output')}
          >
            Output
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="settings-body">
          {/* SECTION 1: AI EVASION */}
          {showEvasion && (
            <div className="settings-section">
              <h4 className="settings-section-title">AI Evasion & Synthesis</h4>

              {/* 1. epsilon */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-epsilon" className="settings-field-label">
                    Perturbation Budget (ε)
                  </label>
                  <span className="settings-value-badge">{params.epsilon.toFixed(1)}</span>
                </div>
                <input
                  id="param-epsilon"
                  type="range"
                  min="0.5"
                  max="64.0"
                  step="0.5"
                  value={params.epsilon}
                  onChange={(e) => updateParam('epsilon', parseFloat(e.target.value))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  L_infinity perturbation budget bound in 8-bit scale [0, 255]. Higher values offer stronger AI evasion.
                </span>
              </div>

              {/* 2. mode */}
              <div className="settings-field">
                <label className="settings-field-label">Synthesis Mode</label>
                <div className="settings-segmented-control" role="radiogroup">
                  {(['hybrid', 'canonical_residual', 'native'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      role="radio"
                      aria-checked={params.mode === m}
                      className={`segment-btn ${params.mode === m ? 'active' : ''}`}
                      onClick={() => updateParam('mode', m)}
                    >
                      {m}
                    </button>
                  ))}
                </div>
                <span className="settings-field-desc">
                  Synthesis mode: 'hybrid' (canonical + high-res tiles), 'canonical_residual' (downscaled global), or 'native' (full-res reflection-padded).
                </span>
              </div>

              {/* 3. refine_steps */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-refine-steps" className="settings-field-label">
                    ViT Refinement Steps
                  </label>
                  <span className="settings-value-badge">{params.refine_steps} steps</span>
                </div>
                <input
                  id="param-refine-steps"
                  type="range"
                  min="0"
                  max="50"
                  step="1"
                  value={params.refine_steps}
                  onChange={(e) => updateParam('refine_steps', parseInt(e.target.value, 10))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  Optional iterative gradient refinement steps against surrogate ViT models.
                </span>
              </div>

              {/* 4. canonical_size */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-canonical-size" className="settings-field-label">
                    Canonical Size
                  </label>
                  <span className="settings-value-badge">{params.canonical_size} px</span>
                </div>
                <input
                  id="param-canonical-size"
                  type="range"
                  min="128"
                  max="1024"
                  step="32"
                  value={params.canonical_size}
                  onChange={(e) => updateParam('canonical_size', parseInt(e.target.value, 10))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  Canonical image dimension for scale-invariant residual synthesis.
                </span>
              </div>

              {/* 5. hybrid_global_weight */}
              {params.mode === 'hybrid' && (
                <div className="settings-field">
                  <div className="settings-field-header">
                    <label htmlFor="param-hybrid-weight" className="settings-field-label">
                      Hybrid Global Weight
                    </label>
                    <span className="settings-value-badge">{params.hybrid_global_weight.toFixed(2)}</span>
                  </div>
                  <input
                    id="param-hybrid-weight"
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={params.hybrid_global_weight}
                    onChange={(e) => updateParam('hybrid_global_weight', parseFloat(e.target.value))}
                    className="settings-slider"
                  />
                  <span className="settings-field-desc">
                    Balancing weight between global thumbnail pass and high-resolution local tile passes in hybrid mode.
                  </span>
                </div>
              )}
            </div>
          )}

          {/* SECTION 2: FEATURE MASKING */}
          {showMasking && (
            <div className="settings-section">
              <h4 className="settings-section-title">Feature Masking & Conformance</h4>

              {/* 6. target_features */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label className="settings-field-label">Target Features</label>
                  <span className="settings-value-badge">
                    {params.target_features.length === 0
                      ? 'Full Image'
                      : params.target_features.join(', ')}
                  </span>
                </div>
                <div className="settings-chips-container">
                  {(['all', 'face', 'person', 'body', 'text'] as const).map((feat) => {
                    const isSelected = params.target_features.includes(feat);
                    return (
                      <button
                        key={feat}
                        type="button"
                        className={`settings-chip ${isSelected ? 'active' : ''}`}
                        onClick={() => toggleFeature(feat)}
                      >
                        {feat}
                      </button>
                    );
                  })}
                  {params.target_features.length > 0 && (
                    <button
                      type="button"
                      className="settings-chip-clear"
                      onClick={() => updateParam('target_features', [])}
                      title="Clear selection to target entire image"
                    >
                      Clear (Entire Image)
                    </button>
                  )}
                </div>
                <span className="settings-field-desc">
                  Comma-separated target silhouette categories (e.g. 'face', 'person', 'text', 'body'). If omitted, applies across entire image.
                </span>
              </div>

              {/* 7. conforming_mask */}
              <label className="settings-toggle-row">
                <span className="toggle-text">
                  <strong>Conforming Mask</strong>
                  <small>
                    When True, bounds perturbations strictly inside detected feature contours with feathered transitions.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={params.conforming_mask}
                  onChange={(e) => updateParam('conforming_mask', e.target.checked)}
                  className="toggle-checkbox"
                  aria-label="Conforming Mask"
                />
              </label>

              {/* 8. feather_radius */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-feather-radius" className="settings-field-label">
                    Feather Radius
                  </label>
                  <span className="settings-value-badge">{params.feather_radius} px</span>
                </div>
                <input
                  id="param-feather-radius"
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={params.feather_radius}
                  onChange={(e) => updateParam('feather_radius', parseInt(e.target.value, 10))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  Gaussian edge-feathering radius in pixels to ensure smooth, invisible mask boundaries.
                </span>
              </div>
            </div>
          )}

          {/* SECTION 3: STEALTH & PRIVACY */}
          {showStealth && (
            <div className="settings-section">
              <h4 className="settings-section-title">Perceptual Stealth & Privacy</h4>

              {/* 9. texture_masking */}
              <label className="settings-toggle-row">
                <span className="toggle-text">
                  <strong>Texture Masking (Weber's Law)</strong>
                  <small>
                    Weber's law contrast-adaptive masking: concentrates noise in high-frequency textured regions and leaves smooth surfaces clean.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={params.texture_masking}
                  onChange={(e) => updateParam('texture_masking', e.target.checked)}
                  className="toggle-checkbox"
                  aria-label="Texture Masking"
                />
              </label>

              {/* 10. chroma_damping */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-chroma-damping" className="settings-field-label">
                    Chroma Damping
                  </label>
                  <span className="settings-value-badge">{params.chroma_damping.toFixed(2)}</span>
                </div>
                <input
                  id="param-chroma-damping"
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={params.chroma_damping}
                  onChange={(e) => updateParam('chroma_damping', parseFloat(e.target.value))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  Opponent chrominance damping factor: suppresses magenta/green chromatic noise shifts to maximize visual stealth.
                </span>
              </div>

              {/* 11. strip_metadata */}
              <label className="settings-toggle-row">
                <span className="toggle-text">
                  <strong>Strip Metadata</strong>
                  <small>
                    Strip EXIF, GPS coordinates, camera serial numbers, and sensitive metadata from the resulting image.
                  </small>
                </span>
                <input
                  type="checkbox"
                  checked={params.strip_metadata}
                  onChange={(e) => updateParam('strip_metadata', e.target.checked)}
                  className="toggle-checkbox"
                  aria-label="Strip Metadata"
                />
              </label>
            </div>
          )}

          {/* SECTION 4: OUTPUT FORMAT */}
          {showOutput && (
            <div className="settings-section">
              <h4 className="settings-section-title">Output & Response</h4>

              {/* 12. output_format */}
              <div className="settings-field">
                <label className="settings-field-label">Output Format</label>
                <div className="settings-segmented-control" role="radiogroup">
                  {(['PNG', 'JPEG', 'WEBP', 'ORIGINAL'] as const).map((fmt) => (
                    <button
                      key={fmt}
                      type="button"
                      role="radio"
                      aria-checked={params.output_format === fmt}
                      className={`segment-btn ${params.output_format === fmt ? 'active' : ''}`}
                      onClick={() => updateParam('output_format', fmt)}
                    >
                      {fmt}
                    </button>
                  ))}
                </div>
                <span className="settings-field-desc">
                  Target image format: 'PNG', 'JPEG', 'WEBP', or 'ORIGINAL'.
                </span>
              </div>

              {/* 13. quality */}
              <div className="settings-field">
                <div className="settings-field-header">
                  <label htmlFor="param-quality" className="settings-field-label">
                    Compression Quality
                  </label>
                  <span className="settings-value-badge">{params.quality}%</span>
                </div>
                <input
                  id="param-quality"
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={params.quality}
                  onChange={(e) => updateParam('quality', parseInt(e.target.value, 10))}
                  className="settings-slider"
                />
                <span className="settings-field-desc">
                  Compression quality factor for lossy formats (JPEG and WEBP).
                </span>
              </div>

              {/* 14. response_type */}
              <div className="settings-field">
                <label className="settings-field-label">Response Type</label>
                <div className="settings-segmented-control" role="radiogroup">
                  {(['image', 'json'] as const).map((res) => (
                    <button
                      key={res}
                      type="button"
                      role="radio"
                      aria-checked={params.response_type === res}
                      className={`segment-btn ${params.response_type === res ? 'active' : ''}`}
                      onClick={() => updateParam('response_type', res)}
                    >
                      {res}
                    </button>
                  ))}
                </div>
                <span className="settings-field-desc">
                  Response payload format: 'image' (binary stream) or 'json' (base64 string + analytics).
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <button
            type="button"
            className="settings-reset-btn"
            onClick={resetDefaults}
            title="Reset all settings to catalog defaults"
          >
            Reset Defaults
          </button>
          <button
            type="button"
            className="settings-action-btn"
            onClick={onClose}
          >
            Apply & Close
          </button>
        </div>
      </div>
    </div>
  );
};
