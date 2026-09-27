import React from 'react';
import { Branding } from './Branding';


export const Sidebar: React.FC = () => {
  return (
    <aside className="app-sidebar" aria-label="Control panel sidebar">
      <Branding />

      {/* Spacer to push lower tweaks section down naturally */}
      <div className="sidebar-spacer" aria-hidden="true" />

     
    </aside>
  );
};
