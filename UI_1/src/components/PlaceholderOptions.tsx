import React, { useState } from 'react';
import { PlusCircleIcon } from './Icons';

interface OptionItem {
  id: string;
  label: string;
  isActivated?: boolean;
}

export const PlaceholderOptions: React.FC = () => {
  const [options, setOptions] = useState<OptionItem[]>([
    { id: 'opt-1', label: 'Add option here', isActivated: false },
    { id: 'opt-2', label: 'Add option here', isActivated: false },
    { id: 'opt-3', label: 'Add option here', isActivated: false },
  ]);

  const [notification, setNotification] = useState<string | null>(null);

  const handleOptionClick = (id: string, index: number) => {
    setOptions((prev) =>
      prev.map((opt) =>
        opt.id === id ? { ...opt, isActivated: !opt.isActivated } : opt
      )
    );

    setNotification(`Option slot ${index + 1} clicked — ready for custom tweak binding.`);
    setTimeout(() => {
      setNotification(null);
    }, 2800);
  };

  return (
    <div className="placeholder-options-section" aria-label="Tweaks and options">
      {/* Top divider separating upper settings from tweaks */}
      <div className="options-divider" role="separator" aria-hidden="true" />

      <h2 className="options-header">TWEAKS / OPTIONS</h2>

      <div className="options-list" role="list">
        {options.map((opt, idx) => (
          <button
            key={opt.id}
            type="button"
            role="listitem"
            className={`option-row-btn ${opt.isActivated ? 'active' : ''}`}
            onClick={() => handleOptionClick(opt.id, idx)}
            aria-label={`${opt.label} (Slot ${idx + 1})`}
          >
            <span className="option-icon" aria-hidden="true">
              <PlusCircleIcon size={19} />
            </span>
            <span className="option-text">
              {opt.isActivated ? `Active: Custom Tweak ${idx + 1}` : opt.label}
            </span>
          </button>
        ))}
      </div>

      {notification && (
        <div className="options-feedback-toast" role="status" aria-live="polite">
          {notification}
        </div>
      )}
    </div>
  );
};
