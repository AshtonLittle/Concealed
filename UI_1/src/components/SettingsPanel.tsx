import React, { useEffect, useRef } from 'react';
import { CloseIcon } from './Icons';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Close on Escape & focus management
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
        <div className="settings-header">
          <div className="settings-title-group">
            <h3 id="settings-title" className="settings-heading">
              SETTINGS
            </h3>
            <span className="settings-subheading">Anti-Analysis Configuration</span>
          </div>
          <button
            ref={closeBtnRef}
            type="button"
            className="settings-close-btn"
            onClick={onClose}
            aria-label="Close settings dialog"
          >
            <CloseIcon size={12} />
          </button>
        </div>

        <div className="settings-body">
          {/* Tweak Group 1 */}
          <div className="settings-group">
            <label className="settings-group-label" id="feathering-label">
              Silhouette Feathering
            </label>
            <div className="settings-segmented-control" role="radiogroup" aria-labelledby="feathering-label">
              <button
                type="button"
                role="radio"
                aria-checked="false"
                className="segment-btn"
              >
                Sharp
              </button>
              <button
                type="button"
                role="radio"
                aria-checked="true"
                className="segment-btn active"
              >
                Smooth
              </button>
              <button
                type="button"
                role="radio"
                aria-checked="false"
                className="segment-btn"
              >
                Soft
              </button>
            </div>
          </div>

          {/* Tweak Group 2: Toggle rows */}
          <div className="settings-toggles">
            <label className="settings-toggle-row">
              <span className="toggle-text">
                <strong>Strip EXIF & Device Metadata</strong>
                <small>Removes sensor tags, camera serial, and GPS coordinates</small>
              </span>
              <input
                type="checkbox"
                defaultChecked
                className="toggle-checkbox"
                aria-label="Strip EXIF and Device Metadata"
              />
            </label>

            <label className="settings-toggle-row">
              <span className="toggle-text">
                <strong>Adversarial Feature Disruption</strong>
                <small>Perturbs embeddings to confuse facial and object detectors</small>
              </span>
              <input
                type="checkbox"
                defaultChecked
                className="toggle-checkbox"
                aria-label="Adversarial Feature Disruption"
              />
            </label>

            <label className="settings-toggle-row">
              <span className="toggle-text">
                <strong>Silhouette Conformance Lock</strong>
                <small>Restricts modifications strictly within isolated contour masks</small>
              </span>
              <input
                type="checkbox"
                defaultChecked
                className="toggle-checkbox"
                aria-label="Silhouette Conformance Lock"
              />
            </label>
          </div>
        </div>

        <div className="settings-footer">
          <button
            type="button"
            className="settings-action-btn"
            onClick={onClose}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
