import React, { useState } from 'react';
import { Download, FileText, Globe, Table, X } from 'lucide-react';
import { RegionPrediction, ModelMetrics } from '../types';
import { exportToCSV, exportToGeoJSON, generatePDFReport } from '../utils/generateReport';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  regions: RegionPrediction[];
  selectedRegion: RegionPrediction;
  metrics?: ModelMetrics;
  chartElementId?: string;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  regions,
  selectedRegion,
  metrics,
  chartElementId,
}) => {
  const [isExportingPDF, setIsExportingPDF] = useState(false);

  if (!isOpen) return null;

  const handleExportPDF = async () => {
    setIsExportingPDF(true);
    try {
      await generatePDFReport(selectedRegion, metrics, chartElementId);
    } finally {
      setIsExportingPDF(false);
    }
  };

  const actionClass = 'w-full flex items-center justify-between gap-3 p-3 bg-[#f3f6f2] hover:bg-[var(--soft-green)] border border-[var(--line)] rounded-lg group transition-colors text-left';
  const titleClass = 'text-xs font-mono font-semibold text-[var(--ink)]';
  const copyClass = 'text-[10px] text-[var(--muted)]';
  const iconClass = 'w-5 h-5';

  return (
    <div className="fixed inset-0 z-50 bg-[#263936]/45 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[var(--surface)] border border-[var(--line)] rounded-xl w-full max-w-md overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-[var(--line)]">
          <div className="flex items-center gap-2">
            <Download className="w-4 h-4 text-[var(--green)]" />
            <h3 className="text-sm font-mono font-semibold text-[var(--ink)]">EXPORT SPEI DATA &amp; REPORTS</h3>
          </div>
          <button onClick={onClose} aria-label="Tutup ekspor" className="text-[var(--muted)] hover:text-[var(--ink)]">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <p className="text-xs text-[var(--muted)] font-mono">
            Select format for public handover / research download:
          </p>

          <div className="space-y-2.5">
            <button onClick={() => { exportToCSV(regions); onClose(); }} className={actionClass}>
              <div className="flex items-center gap-3">
                <Table className={`${iconClass} text-[var(--green)]`} />
                <div>
                  <div className={titleClass}>CSV Format (.csv)</div>
                  <div className={copyClass}>Raw SPEI tabular dataset for all {regions.length} regencies</div>
                </div>
              </div>
              <Download className="w-4 h-4 text-[var(--muted)] group-hover:text-[var(--green)]" />
            </button>

            <button onClick={() => { exportToGeoJSON(regions); onClose(); }} className={actionClass}>
              <div className="flex items-center gap-3">
                <Globe className={`${iconClass} text-[var(--blue)]`} />
                <div>
                  <div className={titleClass}>GeoJSON Format (.geojson)</div>
                  <div className={copyClass}>GIS feature collection with coordinates &amp; metadata</div>
                </div>
              </div>
              <Download className="w-4 h-4 text-[var(--muted)] group-hover:text-[var(--blue)]" />
            </button>

            <button onClick={handleExportPDF} disabled={isExportingPDF} className={`${actionClass} disabled:opacity-50`}>
              <div className="flex items-center gap-3">
                <FileText className={`${iconClass} text-[var(--amber)]`} />
                <div>
                  <div className={titleClass}>PDF Executive Report (.pdf)</div>
                  <div className={copyClass}>Print-ready report for {selectedRegion.regencyName} with embedded chart</div>
                </div>
              </div>
              {isExportingPDF ? (
                <span className="text-[10px] font-mono text-[var(--amber)] animate-pulse">Generating...</span>
              ) : (
                <Download className="w-4 h-4 text-[var(--muted)] group-hover:text-[var(--amber)]" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
