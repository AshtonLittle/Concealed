import React from 'react';

interface BenchmarkViewProps {
  onBack: () => void;
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({ onBack }) => {
  return (
    <div className="benchmark-page" aria-label="Benchmark Evaluation Page">
      {/* Benchmark Header Bar */}
      <div className="benchmark-header">
        <div className="benchmark-title-group">
          <button
            type="button"
            className="benchmark-back-btn"
            onClick={onBack}
            aria-label="Back to main workspace"
          >
            ← Back to Workspace
          </button>
          <h1 className="benchmark-heading">BENCHMARK</h1>
          <p className="benchmark-subheading">
            Anti-Analysis Performance & Detection Resistance Metrics
          </p>
        </div>
        <div className="benchmark-badge-container">
          <span className="benchmark-live-badge">LIVE TEST RUN #410</span>
        </div>
      </div>

      {/* Main Benchmark Metrics Grid */}
      <div className="benchmark-grid">
        {/* Metric 1 */}
        <div className="benchmark-card">
          <div className="benchmark-card-header">
            <span className="benchmark-card-label">FACE DE-IDENTIFICATION</span>
            <span className="benchmark-card-tag">ArcFace / InsightFace</span>
          </div>
          <div className="benchmark-metric-value">99.4%</div>
          <p className="benchmark-metric-desc">
            Suppression rate against deep convolutional feature extractors.
          </p>
          <div className="benchmark-bar-track">
            <div className="benchmark-bar-fill" style={{ width: '99.4%' }} />
          </div>
        </div>

        {/* Metric 2 */}
        <div className="benchmark-card">
          <div className="benchmark-card-header">
            <span className="benchmark-card-label">CONFORMING MASK IoU</span>
            <span className="benchmark-card-tag">YOLO11m + FastSAM</span>
          </div>
          <div className="benchmark-metric-value">0.962</div>
          <p className="benchmark-metric-desc">
            Silhouette overlap ratio conforming strictly to feature contours.
          </p>
          <div className="benchmark-bar-track">
            <div className="benchmark-bar-fill" style={{ width: '96.2%' }} />
          </div>
        </div>

        {/* Metric 3 */}
        <div className="benchmark-card">
          <div className="benchmark-card-header">
            <span className="benchmark-card-label">OPEN-VOCABULARY EVASION</span>
            <span className="benchmark-card-tag">YOLO-World + CLIP</span>
          </div>
          <div className="benchmark-metric-value">98.1%</div>
          <p className="benchmark-metric-desc">
            Competitive suppression against arbitrary text-prompted detectors.
          </p>
          <div className="benchmark-bar-track">
            <div className="benchmark-bar-fill" style={{ width: '98.1%' }} />
          </div>
        </div>

        {/* Metric 4 */}
        <div className="benchmark-card">
          <div className="benchmark-card-header">
            <span className="benchmark-card-label">PIPELINE LATENCY</span>
            <span className="benchmark-card-tag">End-to-End</span>
          </div>
          <div className="benchmark-metric-value">138ms</div>
          <p className="benchmark-metric-desc">
            Full conforming segmentation, perturbation, and re-composition.
          </p>
          <div className="benchmark-bar-track">
            <div className="benchmark-bar-fill" style={{ width: '84%' }} />
          </div>
        </div>
      </div>

      {/* Evaluation Table */}
      <div className="benchmark-table-card">
        <h2 className="benchmark-table-title">MODEL SUPPRESSION COMPARISON</h2>
        <div className="benchmark-table-responsive">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th scope="col">Detector / Architecture</th>
                <th scope="col">Target Class</th>
                <th scope="col">Raw Detection</th>
                <th scope="col">Post-Concealed</th>
                <th scope="col">Resistance Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>ArcFace (ResNet-100)</td>
                <td>Facial Embedding</td>
                <td>0.89 Sim</td>
                <td>0.12 Sim</td>
                <td className="metric-delta">−86.5%</td>
              </tr>
              <tr>
                <td>YOLO11m-pose</td>
                <td>Human Keypoints</td>
                <td>98.2% Conf</td>
                <td>4.1% Conf</td>
                <td className="metric-delta">−95.8%</td>
              </tr>
              <tr>
                <td>PP-OCRv3</td>
                <td>Text & Word Boundaries</td>
                <td>97.5% Acc</td>
                <td>0.0% Acc</td>
                <td className="metric-delta">−100.0%</td>
              </tr>
              <tr>
                <td>YOLOv8x-worldv2</td>
                <td>Arbitrary Objects</td>
                <td>94.7% Conf</td>
                <td>8.3% Conf</td>
                <td className="metric-delta">−91.2%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
