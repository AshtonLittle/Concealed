import { useState } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { VideoWorkspace } from './components/VideoWorkspace';
import type { AppView } from './components/WindowHeader';
import './App.css';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('image');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);

  return (
    <WindowFrame
      currentView={currentView}
      onNavigate={setCurrentView}
      isSidebarOpen={isSidebarOpen}
      onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
      isSettingsOpen={isSettingsOpen}
      onToggleSettings={() => setIsSettingsOpen((prev) => !prev)}
    >
      {/* Left Column: Statistics */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen((prev) => !prev)}
        currentView={currentView === 'video' ? 'video' : 'image'}
      />

      {/* Middle Column: Centered Workspace */}
      {currentView === 'image' && <MainWorkspace />}
      {currentView === 'video' && <VideoWorkspace />}

      {/* Right Column: Settings (Mirrors Left Column) */}
      <SettingsSidebar
        isOpen={isSettingsOpen}
        onToggle={() => setIsSettingsOpen((prev) => !prev)}
        currentView={currentView === 'video' ? 'video' : 'image'}
      />
    </WindowFrame>
  );
}

export default App;
