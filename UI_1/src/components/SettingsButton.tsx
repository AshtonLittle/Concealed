import React from 'react';

interface SettingsButtonProps {
  isOpen: boolean;
  onClick: () => void;
}

export const SettingsButton: React.FC<SettingsButtonProps> = ({ isOpen, onClick }) => {
  return (
    <button
      type="button"
      className={`ios-settings-button ${isOpen ? 'active' : ''}`}
      onClick={onClick}
      aria-label="Application Settings"
      aria-haspopup="dialog"
      aria-expanded={isOpen}
      title="Application Settings"
    >
      <img
        src="/settings.png"
        alt="Application Settings"
        className="settings-icon-img"
      />
    </button>
  );
};
