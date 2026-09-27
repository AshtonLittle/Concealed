import React from 'react';
import type { AppView } from './WindowHeader';

export type ProtectionMode = 'HYBRID' | 'RESIDUAL' | 'NATIVE';
export type ImageOutputFormat = 'PNG' | 'JPEG' | 'WEBP';
export type VideoOutputFormat = 'MP4' | 'WEBM' | 'MOV';

export interface SettingsSidebarProps {
  isOpen?: boolean;
  onToggle?: () => void;
  currentView?: AppView;
  selectedImageModel?: 'onnx' | 'pt';
  onSelectImageModel?: (model: 'onnx' | 'pt') => void;
  budget?: number;
  onBudgetChange?: (budget: number) => void;
  mode?: ProtectionMode;
  onModeChange?: (mode: ProtectionMode) => void;
  format?: string;
  onFormatChange?: (format: string) => void;
}

export const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  isOpen = true,
  onToggle,
  currentView = 'image',
  selectedImageModel = 'onnx',
  onSelectImageModel,
  budget = 8,
  onBudgetChange,
  mode = 'HYBRID',
  onModeChange,
  format = currentView === 'video' ? 'MP4' : 'PNG',
  onFormatChange,
}) => {
  const isVideo = currentView === 'video';

  const formatOptions = isVideo
    ? ['MP4', 'WEBM', 'MOV']
    : ['PNG', 'JPEG', 'WEBP'];

  const modeOptions: ProtectionMode[] = ['HYBRID', 'RESIDUAL', 'NATIVE'];

  return (
    <aside
      className={`app-sidebar settings-sidebar ${!isOpen ? 'collapsed' : ''}`}
      aria-label="Settings sidebar"
      aria-hidden={!isOpen}
    >
      {/* Settings Header & Collapse Control */}
      <div className="sidebar-top-bar">
        <h2 className="sidebar-statistics-heading">SETTINGS</h2>
        {onToggle && (
          <button
            type="button"
            className="sidebar-collapse-btn"
            onClick={onToggle}
            title="Hide Settings Panel"
            aria-label="Hide Settings Panel"
          >
            <svg
              className="collapse-icon-svg"
              viewBox="0 0 24 24"
              width="14"
              height="14"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        )}
      </div>

      <div className="branding-divider" role="separator" aria-hidden="true" />

      {/* Settings Cards */}
      <div className="sidebar-statistics-cards">
        {/* Card 0: Model Engine (ONNX vs PyTorch) */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">MODEL ENGINE</span>
            <span className={`setting-model-status-badge ${isVideo ? 'default-badge' : 'active-tag'}`}>
              {isVideo ? 'DEFAULT (60 FPS)' : (selectedImageModel === 'onnx' ? 'ONNX RUNTIME' : 'PYTORCH')}
            </span>
          </div>

          {isVideo ? (
            <>
              <div className="setting-pill-options">
                <button
                  type="button"
                  className="setting-option-btn active"
                  title="Locked to ONNX Runtime for real-time 60 FPS video frame processing"
                >
                  ONNX (.onnx)
                </button>
                <button
                  type="button"
                  className="setting-option-btn disabled"
                  disabled
                  title="PyTorch .pt is restricted on video to guarantee 60 FPS"
                >
                  PyTorch (.pt)
                </button>
              </div>
              <p className="setting-engine-note">
                ⚡ <strong>generator.onnx</strong> is the default engine for real-time video frame obfuscation.
              </p>
            </>
          ) : (
            <>
              <div className="setting-pill-options">
                <button
                  type="button"
                  className={`setting-option-btn ${selectedImageModel === 'onnx' ? 'active' : ''}`}
                  onClick={() => onSelectImageModel?.('onnx')}
                  title="High-performance ONNX Runtime (generator.onnx)"
                >
                  ONNX (.onnx)
                </button>
                <button
                  type="button"
                  className={`setting-option-btn ${selectedImageModel === 'pt' ? 'active' : ''}`}
                  onClick={() => onSelectImageModel?.('pt')}
                  title="Native PyTorch generator weights (best_generator.pt)"
                >
                  PyTorch (.pt)
                </button>
              </div>
              <p className="setting-engine-note">
                {selectedImageModel === 'onnx'
                  ? '🚀 ONNX Runtime: Fastest execution (<45ms) with optimized graph.'
                  : '🧠 PyTorch: Native ResNet-UNet model (best_generator.pt).'}
              </p>
            </>
          )}
        </div>

        {/* Card 1: Output Format */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">OUTPUT FORMAT</span>
            <span className="setting-model-status-badge active-tag">
              .{format.toLowerCase()}
            </span>
          </div>
          <div className="setting-pill-options">
            {formatOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`setting-option-btn ${format === opt ? 'active' : ''}`}
                onClick={() => onFormatChange?.(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Card 2: Protection Mode */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">PROTECTION MODE</span>
            <span className="setting-model-status-badge active-tag">
              {mode}
            </span>
          </div>
          <div className="setting-pill-options">
            {modeOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`setting-option-btn ${mode === opt ? 'active' : ''}`}
                onClick={() => onModeChange?.(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Card 3: Perturbation Budget */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">PERTURBATION BUDGET (ε)</span>
            <span className="setting-model-status-badge active-tag">
              ε = {budget}
            </span>
          </div>
          <div className="setting-slider-container">
            <div className="setting-slider-track-wrap">
              <input
                type="range"
                min="1"
                max="16"
                step="1"
                value={budget}
                onChange={(e) => onBudgetChange?.(Number(e.target.value))}
                className="setting-budget-slider"
                aria-label="Perturbation budget epsilon slider"
              />
            </div>
            {/* Graduated numbers along the line: 1 to 16 */}
            <div className="setting-slider-scale">
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((num) => {
                const isSelected = Math.round(Number(budget)) === num;
                return (
                  <button
                    key={num}
                    type="button"
                    className={`slider-scale-tick ${isSelected ? 'active' : ''}`}
                    onClick={() => onBudgetChange?.(num)}
                    title={`Set perturbation budget to ${num}`}
                  >
                    <span className="tick-pip" />
                    <span className="tick-label">{num}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
