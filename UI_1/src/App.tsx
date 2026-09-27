import { useState } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import type { ConcealStats } from './components/Sidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { VideoWorkspace } from './components/VideoWorkspace';
import { FeatureObscuringWorkspace } from './components/FeatureObscuringWorkspace';
import type { AppView } from './components/WindowHeader';
import './App.css';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('image');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);
  const [concealStats, setConcealStats] = useState<ConcealStats | null>(null);

  const isFeatureObscuring = currentView === 'feature-obscuring';

  return (
    <WindowFrame
      currentView={currentView}
      onNavigate={setCurrentView}
      isSidebarOpen={isSidebarOpen}
      onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
      isSettingsOpen={isSettingsOpen}
      onToggleSettings={() => setIsSettingsOpen((prev) => !prev)}
    >
      {/* Left Column: Statistics (hidden for feature obscuring) */}
      {!isFeatureObscuring && (
        <Sidebar
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen((prev) => !prev)}
          currentView={currentView}
          stats={concealStats}
        />
      )}

      {/* Middle Column: Centered Workspace */}
      {currentView === 'image' && <MainWorkspace onStatsUpdate={setConcealStats} />}
      {currentView === 'feature-obscuring' && <FeatureObscuringWorkspace />}
      {currentView === 'video' && <VideoWorkspace onStatsUpdate={setConcealStats} />}

      {/* Right Column: Settings (hidden for feature obscuring) */}
      {!isFeatureObscuring && (
        <SettingsSidebar
          isOpen={isSettingsOpen}
          onToggle={() => setIsSettingsOpen((prev) => !prev)}
          currentView={currentView}
        />
      )}
    </WindowFrame>
  );
}

export default App;
