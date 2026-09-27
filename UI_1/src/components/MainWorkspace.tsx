import React from 'react';
import { ProtectionStatus } from './ProtectionStatus';

interface MainWorkspaceProps {
  selectedModelEngine?: 'onnx' | 'pt';
}

export const MainWorkspace: React.FC<MainWorkspaceProps> = ({ selectedModelEngine = 'onnx' }) => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      <div className="workspace-content">
        <ProtectionStatus selectedModelEngine={selectedModelEngine} />
      </div>
    </main>
  );
};
