import React, { useState } from 'react';

interface SettingsSidebarProps {
  isOpen?: boolean;
  onToggle?: () => void;
  currentView?: 'image' | 'video';
}

export const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  isOpen = true,
  onToggle,
  currentView = 'image',
}) => {
  const [resolution, setResolution] = useState(currentView === 'video' ? '1080P' : '1080P');
  const [format, setFormat] = useState(currentView === 'video' ? 'MP4' : 'PNG');
  const [mode, setMode] = useState('HYBRID');
  const [budget, setBudget] = useState('8');

  const resolutionOptions = currentView === 'video'
    ? ['720P', '1080P', '4K']
    : ['1080P', '1440P', '2160P'];

  const formatOptions = currentView === 'video'
    ? ['MP4', 'WEBM', 'MOV']
    : ['PNG', 'JPEG', 'WEBP'];

  const modeOptions = ['HYBRID', 'RESIDUAL', 'NATIVE'];

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

      {/* 4 Clean Settings Cards - Mirroring Left Column Formatting */}
      <div className="sidebar-statistics-cards">
        {/* Card 1: Resolution */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">RESOLUTION</span>
          </div>
          <div className="setting-pill-options">
            {resolutionOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`setting-option-btn ${resolution === opt ? 'active' : ''}`}
                onClick={() => setResolution(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Card 2: Output Format */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">OUTPUT FORMAT</span>
          </div>
          <div className="setting-pill-options">
            {formatOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`setting-option-btn ${format === opt ? 'active' : ''}`}
                onClick={() => setFormat(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Card 3: Protection Mode */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">PROTECTION MODE</span>
          </div>
          <div className="setting-pill-options">
            {modeOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`setting-option-btn ${mode === opt ? 'active' : ''}`}
                onClick={() => setMode(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Card 4: Perturbation Budget */}
        <div className="statistic-card setting-card">
          <div className="setting-card-header">
            <span className="statistic-card-label">PERTURBATION BUDGET (ε)</span>
          </div>
          <div className="setting-slider-container">
            <div className="setting-slider-track-wrap">
              <input
                type="range"
                min="1"
                max="16"
                step="1"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
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
                    onClick={() => setBudget(num.toString())}
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
