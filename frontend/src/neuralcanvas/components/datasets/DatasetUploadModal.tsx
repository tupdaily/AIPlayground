"use client";

import { useState, useCallback, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Upload, X, FileSpreadsheet, Image } from "lucide-react";
import { uploadDataset, type DatasetInfo } from "@/neuralcanvas/lib/trainingApi";

const MAX_FILE_SIZE_MB = 200;

interface DatasetUploadModalProps {
  open: boolean;
  onClose: () => void;
  onUploaded: (dataset: DatasetInfo) => void;
  accessToken: string;
}

export function DatasetUploadModal({ open, onClose, onUploaded, accessToken }: DatasetUploadModalProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [labelColumn, setLabelColumn] = useState("");
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const isCSV = file?.name.toLowerCase().endsWith(".csv");
  const isZip = file?.name.toLowerCase().endsWith(".zip");

  const reset = useCallback(() => {
    setFile(null);
    setName("");
    setLabelColumn("");
    setCsvHeaders([]);
    setError(null);
    setUploading(false);
  }, []);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [onClose, reset]);

  const handleFileChange = useCallback(async (f: File | null) => {
    setError(null);
    setCsvHeaders([]);
    setLabelColumn("");

    if (!f) {
      setFile(null);
      return;
    }

    // Validate extension
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (ext !== "csv" && ext !== "zip") {
      setError("Only .csv and .zip files are supported");
      return;
    }

    // Validate size
    if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Max is ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }

    setFile(f);
    setName(f.name.replace(/\.[^.]+$/, ""));

    // For CSV, parse the first line to get column headers
    if (ext === "csv") {
      try {
        const text = await f.slice(0, 8192).text();
        const firstLine = text.split("\n")[0];
        if (firstLine) {
          const headers = firstLine.split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));
          setCsvHeaders(headers);
          // Default to last column
          if (headers.length > 0) setLabelColumn(headers[headers.length - 1]);
        }
      } catch {
        // Not critical if header parsing fails
      }
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handleFileChange(f);
    },
    [handleFileChange],
  );

  const handleSubmit = useCallback(async () => {
    if (!file || !name.trim()) return;

    setError(null);
    setUploading(true);

    try {
      const dataset = await uploadDataset(
        file,
        name.trim(),
        accessToken,
        isCSV && labelColumn ? labelColumn : undefined,
      );
      onUploaded(dataset);
      handleClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [file, name, accessToken, isCSV, labelColumn, onUploaded, handleClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/50" onClick={handleClose} />

          {/* Panel */}
          <motion.div
            className="relative w-full max-w-md mx-4 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-2xl"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ type: "spring", duration: 0.3 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border)]">
              <h2 className="text-base font-semibold text-[var(--foreground)]">Upload Dataset</h2>
              <button
                onClick={handleClose}
                className="p-1 rounded-md hover:bg-[var(--surface-elevated)] text-[var(--foreground-muted)] transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            {/* Body */}
            <div className="px-5 py-4 space-y-4">
              {/* Drop zone */}
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => inputRef.current?.click()}
                className="flex flex-col items-center gap-2 p-6 rounded-lg border-2 border-dashed border-[var(--border)] hover:border-[var(--accent-muted)] cursor-pointer transition-colors bg-[var(--background)]"
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept=".csv,.zip"
                  className="hidden"
                  onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
                />
                {file ? (
                  <div className="flex items-center gap-2 text-sm text-[var(--foreground)]">
                    {isCSV ? <FileSpreadsheet size={20} className="text-green-500" /> : <Image size={20} className="text-blue-500" />}
                    <span className="font-mono truncate max-w-[200px]">{file.name}</span>
                    <span className="text-[var(--foreground-muted)]">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
                  </div>
                ) : (
                  <>
                    <Upload size={24} className="text-[var(--foreground-muted)]" />
                    <p className="text-sm text-[var(--foreground-muted)]">
                      Drop a <strong>.csv</strong> or <strong>.zip</strong> file here, or click to browse
                    </p>
                    <p className="text-xs text-[var(--foreground-muted)] opacity-60">Max {MAX_FILE_SIZE_MB} MB</p>
                  </>
                )}
              </div>

              {/* Name */}
              {file && (
                <div>
                  <label className="block text-xs font-medium text-[var(--foreground-muted)] mb-1">Dataset Name</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] text-sm focus:outline-none focus:ring-1 focus:ring-[var(--accent-muted)]"
                    placeholder="My Dataset"
                  />
                </div>
              )}

              {/* Label column selector for CSV */}
              {isCSV && csvHeaders.length > 0 && (
                <div>
                  <label className="block text-xs font-medium text-[var(--foreground-muted)] mb-1">Label Column</label>
                  <select
                    value={labelColumn}
                    onChange={(e) => setLabelColumn(e.target.value)}
                    className="w-full px-3 py-2 rounded-md border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] text-sm focus:outline-none focus:ring-1 focus:ring-[var(--accent-muted)]"
                  >
                    {csvHeaders.map((h) => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                  <p className="text-xs text-[var(--foreground-muted)] mt-1 opacity-60">
                    All other columns will be used as features
                  </p>
                </div>
              )}

              {/* Zip hint */}
              {isZip && (
                <p className="text-xs text-[var(--foreground-muted)] opacity-70">
                  Zip should follow the ImageFolder convention: one folder per class, each containing images.
                </p>
              )}

              {/* Error */}
              {error && (
                <p className="text-sm text-red-500 bg-red-500/10 rounded-md px-3 py-2">{error}</p>
              )}
            </div>

            {/* Footer */}
            <div className="flex justify-end gap-2 px-5 py-3 border-t border-[var(--border)]">
              <button
                onClick={handleClose}
                className="px-4 py-2 text-sm rounded-md text-[var(--foreground-muted)] hover:bg-[var(--surface-elevated)] transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={!file || !name.trim() || uploading}
                className="px-4 py-2 text-sm font-medium rounded-md bg-[var(--accent)] text-white hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
              >
                {uploading ? "Uploading..." : "Upload"}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
