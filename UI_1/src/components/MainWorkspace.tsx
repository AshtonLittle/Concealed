import React from 'react';
import { SettingsButton } from './SettingsButton';
import { SettingsPanel } from './SettingsPanel';
import { ProtectionStatus } from './ProtectionStatus';

interface MainWorkspaceProps {
  isSettingsOpen: boolean;
  onToggleSettings: () => void;
  onCloseSettings: () => void;
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({
  isSettingsOpen,
  onToggleSettings,
  onCloseSettings,
}) => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      {/* Top right iOS-style Settings Button */}
      <div className="workspace-top-bar">
        <SettingsButton
          isOpen={isSettingsOpen}
          onClick={onToggleSettings}
        />
      </div>

      {/* Centered Protection Symbol & Typography */}
      <div className="workspace-content">
        <ProtectionStatus />
      </div>

      {/* Settings Dialog / Panel */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={onCloseSettings}
      />
    </main>
  );
};
