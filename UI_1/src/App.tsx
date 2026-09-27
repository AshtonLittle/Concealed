import { useState, useEffect } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import type { ConcealStats } from './components/Sidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { VideoWorkspace } from './components/VideoWorkspace';
import { FeatureObscuringWorkspace } from './components/FeatureObscuringWorkspace';
import { ProbeWorkspace } from './components/ProbeWorkspace';
import type { AppView } from './components/WindowHeader';
import './App.css';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('image');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedImageModel, setSelectedImageModel] = useState<'onnx' | 'pt'>('onnx');
  const [concealStats, setConcealStats] = useState<ConcealStats | null>(null);

  const isFeatureObscuring = currentView === 'feature-obscuring';

  // When switching to video view, collapse settings and sidebar so video fits majority of screen
  useEffect(() => {
    if (currentView === 'video') {
      setIsSettingsOpen(false);
      setIsSidebarOpen(false);
    } else {
      setIsSidebarOpen(true);
    }
  }, [currentView]);

  const handleVideoProcessingStart = () => {
    setIsSettingsOpen(false);
    setIsSidebarOpen(false);
  };

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
      {currentView === 'image' && <MainWorkspace selectedModelEngine={selectedImageModel} onStatsUpdate={setConcealStats} />}
      {currentView === 'feature-obscuring' && <FeatureObscuringWorkspace />}
      {currentView === 'video' && <VideoWorkspace onProcessingStart={handleVideoProcessingStart} onStatsUpdate={setConcealStats} />}
      {currentView === 'probe' && <ProbeWorkspace selectedModelEngine={selectedImageModel} />}

      {/* Right Column: Settings (hidden for feature obscuring) */}
      {!isFeatureObscuring && (
        <SettingsSidebar
          isOpen={isSettingsOpen}
          onToggle={() => setIsSettingsOpen((prev) => !prev)}
          currentView={currentView}
          selectedImageModel={selectedImageModel}
          onSelectImageModel={setSelectedImageModel}
        />
      )}
    </WindowFrame>
  );
}

export default App;
