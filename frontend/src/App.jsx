import { Activity, ExternalLink, FileImage, Layers, Loader2, Upload } from "lucide-react";
import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function artifactUrl(path) {
  if (!path) {
    return "";
  }
  return `${apiUrl}/artifacts/${path}`;
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const reportUrl = result ? `${apiUrl}/analysis/${result.analysis_id}/report` : "";
  const logsUrl = result ? `${apiUrl}/analysis/${result.analysis_id}/logs` : "";
  const visualArtifacts = result
    ? [
        {
          label: "Original",
          src: artifactUrl(result.artifacts.original_image),
        },
        {
          label: "Nucleus mask",
          src: artifactUrl(result.artifacts.mask_image),
        },
        {
          label: "Overlay",
          src: artifactUrl(result.artifacts.overlay_image),
        },
      ]
    : [];

  async function runAnalysis(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose an image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${apiUrl}/analysis`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.message ?? "Analysis failed.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="side-panel">
          <div className="brand-row">
            <Activity size={24} />
            <span>Neutrophil Analysis</span>
          </div>
          <form className="upload-panel" onSubmit={runAnalysis}>
            <label className="file-zone">
              <FileImage size={30} />
              <span>{file ? file.name : "Drop or choose a single-cell image"}</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/tiff,image/bmp"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {previewUrl && (
              <figure className="preview-card">
                <img src={previewUrl} alt="Selected cell preview" />
                <figcaption>Selected image</figcaption>
              </figure>
            )}
            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
              <span>{isLoading ? "Analyzing" : "Run analysis"}</span>
            </button>
            {error && <p className="error-text">{error}</p>}
          </form>
        </aside>

        <section className="result-panel">
          <header className="result-header">
            <h1>Single-cell nucleus segmentation</h1>
            <span className="status-pill">{result?.status ?? "idle"}</span>
          </header>

          {result ? (
            <>
              <div className="result-grid">
                <div className="metric">
                  <span>Classification</span>
                  <strong>{result.classification.label}</strong>
                </div>
                <div className="metric">
                  <span>Nucleus segments</span>
                  <strong>{result.features.nucleus_segments}</strong>
                </div>
                <div className="metric">
                  <span>Nucleus area</span>
                  <strong>{result.features.nucleus_area_px}px</strong>
                </div>
                <div className="metric">
                  <span>Confidence</span>
                  <strong>{Math.round(result.classification.confidence * 100)}%</strong>
                </div>
              </div>

              <section className="visual-grid" aria-label="Analysis artifacts">
                {visualArtifacts.map((artifact) => (
                  <figure className="artifact-card" key={artifact.label}>
                    <div className="artifact-image-frame">
                      <img src={artifact.src} alt={artifact.label} />
                    </div>
                    <figcaption>
                      <Layers size={16} />
                      <span>{artifact.label}</span>
                    </figcaption>
                  </figure>
                ))}
              </section>

              <section className="feature-panel">
                <h2>Key morphology</h2>
                <div className="feature-list">
                  <span>Circularity</span>
                  <strong>{result.features.nucleus_circularity.toFixed(3)}</strong>
                  <span>Solidity</span>
                  <strong>{result.features.nucleus_solidity.toFixed(3)}</strong>
                  <span>Aspect ratio</span>
                  <strong>{result.features.nucleus_aspect_ratio.toFixed(3)}</strong>
                  <span>Foreground fraction</span>
                  <strong>{result.features.mask_foreground_fraction.toFixed(3)}</strong>
                </div>
                <p>{result.classification.reason}</p>
                <div className="report-links">
                  <a href={reportUrl} target="_blank" rel="noreferrer">
                    <ExternalLink size={15} />
                    Report
                  </a>
                  <a href={logsUrl} target="_blank" rel="noreferrer">
                    <ExternalLink size={15} />
                    Logs
                  </a>
                </div>
              </section>

              <pre className="json-view">{JSON.stringify(result, null, 2)}</pre>
            </>
          ) : (
            <div className="empty-state">
              <Activity size={36} />
              <p>Upload one neutrophil image to run preprocessing, nucleus segmentation, morphology extraction, and classification.</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
