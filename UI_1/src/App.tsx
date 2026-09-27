import { useState, useEffect } from 'react';
import { WindowFrame } from './components/WindowFrame';
import { Sidebar } from './components/Sidebar';
import type { ConcealStats } from './components/Sidebar';
import { SettingsSidebar } from './components/SettingsSidebar';
import type { ProtectionMode, ImageOutputFormat, VideoOutputFormat } from './components/SettingsSidebar';
import { MainWorkspace } from './components/MainWorkspace';
import { VideoWorkspace } from './components/VideoWorkspace';
import { FeatureObscuringWorkspace } from './components/FeatureObscuringWorkspace';
import { ProbeWorkspace } from './components/ProbeWorkspace';
import type { AppView } from './components/WindowHeader';
import './App.css';

export function App() {
  const [currentView, setCurrentView] = useState<AppView>('image');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);
  const [selectedImageModel, setSelectedImageModel] = useState<'onnx' | 'pt'>('onnx');
  const [budget, setBudget] = useState<number>(8);
  const [mode, setMode] = useState<ProtectionMode>('HYBRID');
  const [imageFormat, setImageFormat] = useState<ImageOutputFormat>('PNG');
  const [videoFormat, setVideoFormat] = useState<VideoOutputFormat>('MP4');
  const [concealStats, setConcealStats] = useState<ConcealStats | null>(null);

  const isNoSidebarView = currentView === 'feature-obscuring' || currentView === 'probe';

  // When switching to video view, collapse settings and sidebar so video fits majority of screen
  useEffect(() => {
    if (currentView === 'video') {
      setIsSettingsOpen(false);
      setIsSidebarOpen(false);
    } else if (currentView !== 'feature-obscuring' && currentView !== 'probe') {
      setIsSidebarOpen(true);
      setIsSettingsOpen(true);
    }
  }, [currentView]);

  const handleVideoProcessingStart = () => {
    setIsSettingsOpen(false);
    setIsSidebarOpen(false);
  };

  const activeFormat = currentView === 'video' ? videoFormat : imageFormat;
  const handleFormatChange = (fmt: string) => {
    if (currentView === 'video') {
      setVideoFormat(fmt as VideoOutputFormat);
    } else {
      setImageFormat(fmt as ImageOutputFormat);
    }
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
      {/* Left Column: Statistics (hidden for feature obscuring and model probe) */}
      {!isNoSidebarView && (
        <Sidebar
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen((prev) => !prev)}
          currentView={currentView}
          stats={concealStats}
        />
      )}

      {/* Middle Column: Centered Workspace */}
      {currentView === 'image' && (
        <MainWorkspace
          selectedModelEngine={selectedImageModel}
          budget={budget}
          mode={mode}
          outputFormat={imageFormat}
          onStatsUpdate={setConcealStats}
        />
      )}
      {currentView === 'feature-obscuring' && <FeatureObscuringWorkspace />}
      {currentView === 'video' && (
        <VideoWorkspace
          budget={budget}
          mode={mode}
          videoFormat={videoFormat}
          onBudgetChange={setBudget}
          onModeChange={setMode}
          onProcessingStart={handleVideoProcessingStart}
          onStatsUpdate={setConcealStats}
        />
      )}
      {currentView === 'probe' && <ProbeWorkspace selectedModelEngine={selectedImageModel} />}

      {/* Right Column: Settings (hidden for feature obscuring and model probe) */}
      {!isNoSidebarView && (
        <SettingsSidebar
          isOpen={isSettingsOpen}
          onToggle={() => setIsSettingsOpen((prev) => !prev)}
          currentView={currentView}
          selectedImageModel={selectedImageModel}
          onSelectImageModel={setSelectedImageModel}
          budget={budget}
          onBudgetChange={setBudget}
          mode={mode}
          onModeChange={setMode}
          format={activeFormat}
          onFormatChange={handleFormatChange}
        />
      )}
    </WindowFrame>
  );
}

export default App;
