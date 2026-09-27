import React from 'react';
import { ProtectionStatus } from './ProtectionStatus';

export const MainWorkspace: React.FC = () => {
  return (
    <main className="main-workspace" aria-label="Main protection workspace">
      <div className="workspace-content">
        <ProtectionStatus />
      </div>
    </main>
  );
};
