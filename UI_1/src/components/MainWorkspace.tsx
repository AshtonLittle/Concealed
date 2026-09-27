import React from 'react';
import { SettingsButton } from './SettingsButton';
import { SettingsPanel, type ObfuscatorParameters } from './SettingsPanel';
import { ProtectionStatus } from './ProtectionStatus';

interface MainWorkspaceProps {
  isSettingsOpen: boolean;
  onToggleSettings: () => void;
  onCloseSettings: () => void;
  parameters?: ObfuscatorParameters;
  onChangeParameters?: (params: ObfuscatorParameters) => void;
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({
  isSettingsOpen,
  onToggleSettings,
  onCloseSettings,
  parameters,
  onChangeParameters,
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

      {/* Centered Protection Symbol, Upload & Result */}
      <div className="workspace-content">
        <ProtectionStatus parameters={parameters} />
      </div>

      {/* Settings Dialog / Panel */}
      <SettingsPanel
        isOpen={isSettingsOpen}
        onClose={onCloseSettings}
        parameters={parameters}
        onChangeParameters={onChangeParameters}
      />
    </main>
  );
};
