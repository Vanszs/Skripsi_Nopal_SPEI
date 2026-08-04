import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Download,
  Droplets,
  FileText,
  Map,
  Menu,
  Search,
  ShieldAlert,
  X,
} from 'lucide-react';
import { DroughtMap } from './components/DroughtMap';
import { ExportModal } from './components/ExportModal';
import { TFTFanChart } from './components/TFTFanChart';
import { DEFAULT_METRICS, fetchStudyData } from './api';
import { DroughtSeverity, RegionPrediction } from './types';
import { MOCK_REGIONS } from './data/mockData';

type ActiveView = 'map' | 'forecast' | 'risk' | 'method';

const severityMeta: Record<DroughtSeverity, { label: string; className: string; action: string }> = {
  NORMAL: {
    label: 'Normal',
    className: 'severity-normal',
    action: 'Pantau rutin dan pertahankan cadangan air untuk periode berikutnya.',
  },
  MODERATE: {
    label: 'Waspada',
    className: 'severity-moderate',
    action: 'Atur irigasi lebih hemat dan mulai menyiapkan sumber air alternatif.',
  },
  SEVERE: {
    label: 'Siaga',
    className: 'severity-severe',
    action: 'Prioritaskan distribusi air, tanaman tahan kering, dan kesiapan pompa irigasi.',
  },
  EXTREME: {
    label: 'Awas',
    className: 'severity-extreme',
    action: 'Aktifkan langkah tanggap kekeringan bersama BPBD, dinas pertanian, dan desa.',
  },
};

const views = [
  { id: 'map' as const, label: 'Peta', icon: Map },
  { id: 'forecast' as const, label: 'Prediksi', icon: Activity },
  { id: 'risk' as const, label: 'Risiko', icon: ShieldAlert },
  { id: 'method' as const, label: 'Metode', icon: BookOpen },
];

const EMPTY_REGION: RegionPrediction = {
  id: 'loading', regencyName: 'Memuat wilayah...', province: 'Jawa Timur',
  speiCurrent: 0, speiForecast: { q10: 0, q50: 0, q90: 0 }, severity: 'NORMAL',
  coordinates: [-7.25, 112.75], historicalSpei: [],
};

export default function App() {
  const [activeView, setActiveView] = useState<ActiveView>('map');
  const [selectedHorizon, setSelectedHorizon] = useState(3);
  const [selectedAreaId, setSelectedAreaId] = useState<string | null>(null);
  const [regions, setRegions] = useState<RegionPrediction[]>(MOCK_REGIONS);
  const [studyGrid, setStudyGrid] = useState<Array<{ id: string; city_id: string; lat: number; lon: number; spei: number; selected_rank?: number }>>(
    MOCK_REGIONS.flatMap((region) => {
      const offsets = [
        { dLat: 0.0, dLon: 0.0 },
        { dLat: -0.08, dLon: -0.08 },
        { dLat: 0.0, dLon: -0.12 },
        { dLat: -0.08, dLon: 0.08 },
        { dLat: 0.08, dLon: -0.08 },
      ];
      return offsets.map((offset, index) => ({
        id: `${region.id}__n${String(index).padStart(2, '0')}__fallback`,
        city_id: region.id,
        lat: region.coordinates[0] + offset.dLat,
        lon: region.coordinates[1] + offset.dLon,
        spei: region.speiForecast.q50,
        selected_rank: index + 1,
      }));
    }),
  );
  const [gridVisible, setGridVisible] = useState(true);
  const [nodesVisible, setNodesVisible] = useState(true);
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);
  const [selectedRegionState, setSelectedRegion] = useState<RegionPrediction | null>(null);
  const [dataStatus, setDataStatus] = useState('DATA DUMMY · menunggu data penelitian');
  const [dataError, setDataError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isExportOpen, setIsExportOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetchStudyData(controller.signal).then((study) => {
      setRegions(study.regions);
      setStudyGrid(study.grid);
      setSelectedRegion(study.regions[0] ?? null);
      setDataStatus(`${study.status} · observasi ${study.observationPeriod[0]}–${study.observationPeriod[1]}`);
    }).catch((error: unknown) => {
      if ((error as Error).name !== 'AbortError') {
        setDataError(error instanceof Error ? error.message : 'Data penelitian gagal dimuat.');
        setDataStatus('Data belum tersedia');
      }
    });
    return () => controller.abort();
  }, []);

  const filteredRegions = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return [];
    return regions.filter((region) =>
      `${region.regencyName} ${region.province}`.toLowerCase().includes(query),
    );
  }, [regions, searchQuery]);

  const selectedRegion = selectedRegionState ?? regions[0] ?? EMPTY_REGION;
  const selectedSeverity = severityMeta[selectedRegion.severity];
  const watchedCount = regions.length;
  const atRiskCount = regions.filter((region) => region.severity !== 'NORMAL').length;

  const selectRegion = (region: RegionPrediction) => {
    setSelectedRegion(region);
    setSearchQuery('');
    setActiveView('map');
  };

  const selectForecastRegion = (region: RegionPrediction) => {
    setSelectedRegion(region);
    setSelectedHorizon(3);
  };

  const selectedGrid = useMemo(() => {
    const keys = new Set([selectedRegion.id.toLowerCase(), selectedRegion.regencyName.toLowerCase().replace(/^kab\.\s*/i, '')]);
    return studyGrid.filter((cell) => keys.has(cell.city_id.toLowerCase().replace(/^kab\.\s*/i, '')));
  }, [selectedRegion, studyGrid]);

  const activeArea = selectedGrid.find((cell) => String(cell.selected_rank ?? '') === selectedAreaId) ?? selectedGrid[0];

  return (
    <div className="drought-app">
      <a className="skip-link" href="#workspace">Langsung ke peta dan analisis</a>

      <aside className="side-rail" aria-label="Navigasi utama">
        <button className="brand-mark" onClick={() => setActiveView('map')} aria-label="NusaPantau, halaman utama">
          <Droplets size={21} strokeWidth={1.8} />
          <span>NP</span>
        </button>
        <nav className="rail-nav">
          {views.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveView(id)}
              className={activeView === id ? 'is-active' : ''}
              aria-current={activeView === id ? 'page' : undefined}
              title={label}
            >
              <Icon size={18} strokeWidth={1.7} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <header className="app-bar">
        <div className="app-identity">
          <button className="mobile-menu" onClick={() => setMobileNavOpen((open) => !open)} aria-label="Buka navigasi" aria-expanded={mobileNavOpen}>
            {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <button className="identity-copy" onClick={() => setActiveView('map')}>
            <strong>NusaPantau Kekeringan</strong>
            <span>Proyeksi SPEI Jawa Timur</span>
          </button>
          <span className="data-badge">DATA PENELITIAN</span>
        </div>

        <div className="app-actions">
          <div className="search-control">
            <Search size={16} aria-hidden="true" />
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Cari kabupaten"
              aria-label="Cari kabupaten"
            />
            {searchQuery && <button onClick={() => setSearchQuery('')} aria-label="Hapus pencarian"><X size={15} /></button>}
            {filteredRegions.length > 0 && (
              <div className="search-results" role="listbox">
                {filteredRegions.map((region) => (
                  <button key={region.id} onClick={() => selectRegion(region)} role="option">
                    <strong>{region.regencyName}</strong>
                    <span>{region.province}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button className="icon-button" onClick={() => setIsExportOpen(true)} title="Unduh laporan" aria-label="Unduh laporan">
            <Download size={18} />
          </button>
        </div>

        {mobileNavOpen && (
          <nav className="mobile-nav" aria-label="Navigasi utama mobile">
            {views.map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => { setActiveView(id); setMobileNavOpen(false); }} className={activeView === id ? 'is-active' : ''}>
                <Icon size={18} /> {label}
              </button>
            ))}
          </nav>
        )}
      </header>

      <main id="workspace" className="workspace">
        {dataError && <div className="data-error" role="alert">{dataError}</div>}
        {!dataError && regions.length === 0 && <div className="data-loading" role="status">{dataStatus}</div>}
        {activeView === 'map' && (
          <section className="map-workspace" aria-label="Peta dan analisis kekeringan">
            <div className="map-stage">
              <DroughtMap regions={regions} grid={studyGrid} gridVisible={gridVisible} nodesVisible={nodesVisible} onToggleGrid={() => setGridVisible((visible) => !visible)} onToggleNodes={() => setNodesVisible((visible) => !visible)} selectedRegion={selectedRegion} onSelectRegion={selectRegion} height="100%" />
              <div className="map-intro">
                <span>PROYEK SKRIPSI</span>
                <h1>Perkiraan risiko kekeringan untuk lima wilayah studi.</h1>
                <p>Pilih titik pada peta untuk membaca proyeksi SPEI dan rentang ketidakpastiannya.</p>
              </div>
              <div className="map-sample-note">{dataStatus}. Bukan peringatan operasional.</div>
            </div>

            <section className={`analysis-drawer ${isAnalysisExpanded ? 'is-expanded' : ''}`} aria-label="Analisis wilayah terpilih">
              <div className="drawer-heading">
                <div>
                  <p>ANALISIS WILAYAH</p>
                  <h2>{selectedRegion.regencyName}</h2>
                  <span>{selectedRegion.province} · {selectedRegion.coordinates[0].toFixed(3)}, {selectedRegion.coordinates[1].toFixed(3)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="horizon-control" aria-label="Pilih horizon prediksi">
                    {[1, 3, 6, 12].map((horizon) => (
                      <button key={horizon} onClick={() => setSelectedHorizon(horizon)} className={selectedHorizon === horizon ? 'is-selected' : ''}>
                        +{horizon} bln
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    aria-expanded={isAnalysisExpanded}
                    aria-label={isAnalysisExpanded ? 'Ringkas analisis' : 'Perluas analisis'}
                    onClick={() => setIsAnalysisExpanded((expanded) => !expanded)}
                    className="drawer-toggle"
                    title={isAnalysisExpanded ? 'Ringkas analisis' : 'Perluas analisis'}
                  >
                    {isAnalysisExpanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                  </button>
                </div>
              </div>

              <div className="analysis-grid">
                <div className="analysis-panel chart-panel">
                  <TFTFanChart region={selectedRegion} selectedHorizon={selectedHorizon} onHorizonChange={setSelectedHorizon} />
                </div>
                <section className="analysis-panel severity-panel">
                  <div className={`severity-status ${selectedSeverity.className}`}>
                    <span>Status +{selectedHorizon} bulan</span>
                    <strong>{selectedSeverity.label}</strong>
                    <b>SPEI median {selectedRegion.speiForecast.q50.toFixed(2)}</b>
                  </div>
                  <p>{selectedSeverity.action}</p>
                  <button onClick={() => setActiveView('risk')}>Buka rekomendasi</button>
                </section>
                <section className="analysis-panel model-panel">
                  <p>VALIDASI MODEL</p>
                  <strong>{(DEFAULT_METRICS.skillScore * 100).toFixed(1)}%</strong>
                  <span>skill score multi-seed</span>
                  <dl>
                    <div><dt>RMSE</dt><dd>{DEFAULT_METRICS.rmse.toFixed(3)}</dd></div>
                    <div><dt>MAE</dt><dd>{DEFAULT_METRICS.mae.toFixed(3)}</dd></div>
                  </dl>
                </section>
              </div>
            </section>
          </section>
        )}

        {activeView === 'forecast' && (
          <section className="standalone-view forecast-view">
            <header><span>PREDIKSI PROBABILISTIK</span><h1>Prediksi lima wilayah studi</h1><p>Pilih kabupaten untuk membaca lintasan SPEI, rentang kemungkinan, dan grid sumber model.</p></header>
            <div className="forecast-layout">
              <aside className="forecast-sidebar" aria-label="Daftar wilayah dan grid penelitian">
                <div className="forecast-sidebar__heading"><strong>WILAYAH STUDI</strong><span>{regions.length} kabupaten</span></div>
                <div className="forecast-region-list">
                  {regions.map((region) => {
                    const active = region.id === selectedRegion.id;
                    const regionGrid = studyGrid.filter((cell) => cell.city_id.toLowerCase().replace(/^kab\.\s*/i, '') === region.id.toLowerCase().replace(/^kab\.\s*/i, ''));
                    const meta = severityMeta[region.severity];
                    return <button key={region.id} type="button" className={`forecast-region ${active ? 'is-selected' : ''}`} onClick={() => selectForecastRegion(region)} aria-pressed={active}>
                      <span className="forecast-region__top"><strong>{region.regencyName.replace(/^Kab\.\s*/i, '')}</strong><b className={meta.className}>{meta.label}</b></span>
                      <span className="forecast-region__meta">SPEI {region.speiForecast.q50.toFixed(2)} · {regionGrid.length || '—'} grid</span>
                    </button>;
                  })}
                </div>
                <div className="forecast-grid-detail">
                  <div className="forecast-sidebar__heading"><strong>GRID {selectedRegion.regencyName.replace(/^Kab\.\s*/i, '').toUpperCase()}</strong><span>{selectedGrid.length} cell</span></div>
                  <div className="forecast-grid-list">
                    {selectedGrid.map((cell) => <button key={cell.id} type="button" className={`forecast-grid-row ${String(cell.selected_rank ?? '') === String(activeArea?.selected_rank ?? '') ? 'is-selected' : ''}`} onClick={() => setSelectedAreaId(String(cell.selected_rank ?? cell.id))} aria-pressed={String(cell.selected_rank ?? '') === String(activeArea?.selected_rank ?? '')}><span><b>{cell.selected_rank ? `Area ${cell.selected_rank}` : cell.id}</b><small>{cell.lat.toFixed(3)}, {cell.lon.toFixed(3)}</small></span><strong>{cell.spei.toFixed(2)}</strong></button>)}
                    {selectedGrid.length === 0 && <p className="forecast-empty">Grid source belum tersedia untuk wilayah ini.</p>}
                  </div>
                </div>
              </aside>
              <div className="standalone-chart"><div className="forecast-area-context"><strong>{activeArea ? `Area ${activeArea.selected_rank ?? 'aktif'}` : 'Area belum dipilih'}</strong><span>Forecast model tersedia pada level kabupaten; nilai area menunjukkan cell sumber {activeArea ? `· SPEI ${activeArea.spei.toFixed(2)}` : ''}.</span></div><TFTFanChart region={selectedRegion} selectedHorizon={selectedHorizon} onHorizonChange={setSelectedHorizon} /></div>
            </div>
          </section>
        )}

        {activeView === 'risk' && (
          <section className="standalone-view risk-view">
            <header><span>RISIKO WILAYAH STUDI</span><h1>Prioritas respons kekeringan</h1><p>Urutkan tindakan dari hasil proyeksi, bukan status peringatan resmi.</p></header>
            <div className="risk-list">
              {regions.map((region) => {
                const meta = severityMeta[region.severity];
                return <button key={region.id} onClick={() => selectRegion(region)} className="risk-row">
                  <div><strong>{region.regencyName}</strong><span>{region.province}</span></div>
                  <span className={`risk-label ${meta.className}`}>{meta.label}</span>
                  <b>{region.speiForecast.q50.toFixed(2)}</b>
                  <span className="risk-action">{meta.action}</span>
                </button>;
              })}
            </div>
          </section>
        )}

        {activeView === 'method' && (
          <section className="standalone-view method-view">
            <header><span>METODOLOGI</span><h1>Dari cuaca ke indeks kekeringan</h1><p>Ringkasan alur penelitian untuk membaca hasil visualisasi secara tepat.</p></header>
            <div className="method-grid">
              <article><Droplets size={22} /><h2>Data cuaca</h2><p>Curah hujan dan evapotranspirasi dari Open-Meteo membentuk neraca air.</p></article>
              <article><Activity size={22} /><h2>Indeks SPEI</h2><p>Neraca air dihitung dalam skala waktu untuk mengukur kondisi kering atau basah.</p></article>
              <article><FileText size={22} /><h2>Model TFT</h2><p>Temporal Fusion Transformer memproyeksikan q10, q50, dan q90 hingga 12 bulan.</p></article>
            </div>
          </section>
        )}

        <footer className="workspace-footer">
          <span>{watchedCount} kabupaten studi</span><span>{atRiskCount} wilayah berada di atas status normal pada contoh ini</span><span>Riset TFT-SPEI · 2026</span>
        </footer>
      </main>

      <ExportModal isOpen={isExportOpen} onClose={() => setIsExportOpen(false)} regions={regions} selectedRegion={selectedRegion} />
    </div>
  );
}