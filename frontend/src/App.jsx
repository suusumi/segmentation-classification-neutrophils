import { Activity, FileImage, Loader2, Upload } from "lucide-react";
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

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
              <pre className="json-view">{JSON.stringify(result, null, 2)}</pre>
            </div>
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
