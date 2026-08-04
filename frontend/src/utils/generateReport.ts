import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
import { RegionPrediction, ModelMetrics } from '../types';

export function exportToCSV(regions: RegionPrediction[]) {
  const headers = ['BPS ID', 'Regency', 'Province', 'Latitude', 'Longitude', 'Current SPEI', 'Forecast q10', 'Forecast q50', 'Forecast q90', 'Severity'];
  const rows = regions.map(r => [
    r.id,
    `"${r.regencyName}"`,
    `"${r.province}"`,
    r.coordinates[0],
    r.coordinates[1],
    r.speiCurrent.toFixed(2),
    r.speiForecast.q10.toFixed(2),
    r.speiForecast.q50.toFixed(2),
    r.speiForecast.q90.toFixed(2),
    r.severity
  ]);

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `SPEI_Forecast_Export_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function exportToGeoJSON(regions: RegionPrediction[]) {
  const geojson = {
    type: 'FeatureCollection',
    features: regions.map(r => ({
      type: 'Feature',
      geometry: {
        type: 'Point',
        coordinates: [r.coordinates[1], r.coordinates[0]] // GeoJSON uses [lng, lat]
      },
      properties: {
        id: r.id,
        regencyName: r.regencyName,
        province: r.province,
        speiCurrent: r.speiCurrent,
        speiForecast: r.speiForecast,
        severity: r.severity
      }
    }))
  };

  const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `SPEI_Forecast_GeoJSON_${new Date().toISOString().slice(0, 10)}.geojson`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export async function generatePDFReport(
  region: RegionPrediction,
  metrics?: ModelMetrics,
  chartElementId?: string
) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  // Header banner
  doc.setFillColor(14, 20, 32);
  doc.rect(0, 0, 210, 30, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.text('INDONESIA HYDRO-METEOROLOGICAL SPEI REPORT', 14, 15);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(148, 163, 184);
  doc.text(`Generated: ${new Date().toLocaleString()} • Public Handover Release`, 14, 23);

  // Region details card
  doc.setFillColor(241, 245, 249);
  doc.rect(14, 38, 182, 40, 'F');
  doc.setDrawColor(203, 213, 225);
  doc.rect(14, 38, 182, 40, 'S');

  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text(`${region.regencyName}, ${region.province}`, 20, 48);

  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(71, 85, 105);
  doc.text(`BPS Region ID: ${region.id}   |   Coordinates: ${region.coordinates[0]}°, ${region.coordinates[1]}°`, 20, 56);
  doc.text(`Current SPEI Index: ${region.speiCurrent.toFixed(2)}   |   Status: ${region.severity}`, 20, 64);
  doc.text(`Quantile Forecast (+3M): q10=${region.speiForecast.q10.toFixed(2)} | q50=${region.speiForecast.q50.toFixed(2)} | q90=${region.speiForecast.q90.toFixed(2)}`, 20, 72);

  let currentY = 86;

  // Render chart if present
  if (chartElementId) {
    const el = document.getElementById(chartElementId);
    if (el) {
      const canvas = await html2canvas(el, { scale: 2, backgroundColor: '#f3f6f2' });
      const imgData = canvas.toDataURL('image/png');
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(15, 23, 42);
      doc.text('TFT Forecast Fan-Chart', 14, currentY);
      currentY += 4;
      doc.addImage(imgData, 'PNG', 14, currentY, 182, 80);
      currentY += 86;
    }
  }

  // Model Verification Metrics table
  if (metrics) {
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(15, 23, 42);
    doc.text('Model Verification & Quality Metrics', 14, currentY);
    currentY += 6;

    const tableData = [
      ['Quantile Loss (Pinball)', String(metrics.quantileLoss)],
      ['RMSE / MAE', `${metrics.rmse} / ${metrics.mae}`],
      ['Skill Score', `${(metrics.skillScore * 100).toFixed(1)}%`],
      ['Dropout Rate', String(metrics.dropoutRate)],
      ['Multi-Seed Status', metrics.multiSeedVerified ? 'VERIFIED (3/3 passed)' : 'UNVERIFIED'],
      ['Git Audit Commit', metrics.commitHash]
    ];

    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    tableData.forEach(([label, val]) => {
      doc.setFillColor(248, 250, 252);
      doc.rect(14, currentY, 90, 7, 'F');
      doc.rect(104, currentY, 92, 7, 'F');
      doc.setDrawColor(226, 232, 240);
      doc.rect(14, currentY, 90, 7, 'S');
      doc.rect(104, currentY, 92, 7, 'S');
      
      doc.setTextColor(100, 116, 139);
      doc.text(label, 18, currentY + 5);
      doc.setTextColor(15, 23, 42);
      doc.text(val, 108, currentY + 5);
      currentY += 7;
    });
  }

  // Footer
  doc.setFontSize(8);
  doc.setTextColor(148, 163, 184);
  doc.text('Temporal Fusion Transformer SPEI Forecasting System • Automated Public Handover PDF', 14, 285);

  doc.save(`SPEI_Report_${region.id}_${new Date().toISOString().slice(0, 10)}.pdf`);
}
