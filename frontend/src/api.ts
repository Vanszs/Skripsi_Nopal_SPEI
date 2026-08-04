import { RegionPrediction, ModelMetrics } from './types';
import { MOCK_REGIONS } from './data/mockData';

interface StudyResponse {
  data_status: string;
  source: string;
  observation_period: string[];
  prediction_period: string[];
  regions: Array<{
    id: string;
    regency_name: string;
    province: string;
    coordinates: [number, number];
    spei_current: number;
    severity: RegionPrediction['severity'];
    latest_observation: string;
    evaluation_prediction: { p10: number; p50: number; p90: number } | null;
    historical_spei: Array<{ month: string; actual: number; predicted: number }>;
  }>;
  grid: Array<{ id: string; city_id: string; lat: number; lon: number; spei: number; selected_rank?: number }>;
}

export interface StudyData {
  regions: RegionPrediction[];
  status: string;
  source: string;
  observationPeriod: string[];
  predictionPeriod: string[];
  grid: StudyResponse['grid'];
}

export const DEFAULT_METRICS: ModelMetrics = {
  quantileLoss: 0,
  rmse: 0.438,
  mae: 0.314,
  skillScore: 0.199,
  dropoutRate: 0.40,
  multiSeedVerified: true,
  commitHash: 'evaluation artifact 20260602',
};

export async function fetchStudyData(signal?: AbortSignal): Promise<StudyData> {
  try {
    const response = await fetch('/api/v1/study/regions', { signal });
    if (!response.ok) throw new Error(`Gagal memuat data penelitian (${response.status})`);
    const payload = (await response.json()) as StudyResponse;
    return mapPayload(payload);
  } catch (error) {
    if ((error as Error).name === 'AbortError') throw error;
    console.warn('Backend tidak tersedia, fallback ke data dummy:', error);
    return fallbackStudyData();
  }
}

function mapPayload(payload: StudyResponse): StudyData {
  return {
    status: payload.data_status,
    source: payload.source,
    observationPeriod: payload.observation_period,
    predictionPeriod: payload.prediction_period,
    grid: payload.grid,
    regions: payload.regions.map((region) => ({
      id: region.id,
      regencyName: region.regency_name,
      province: region.province,
      speiCurrent: region.spei_current,
      speiForecast: {
        q10: region.evaluation_prediction?.p10 ?? region.spei_current,
        q50: region.evaluation_prediction?.p50 ?? region.spei_current,
        q90: region.evaluation_prediction?.p90 ?? region.spei_current,
      },
      severity: region.severity,
      coordinates: region.coordinates,
      historicalSpei: region.historical_spei,
    })),
  };
}

function fallbackStudyData(): StudyData {
  const grid = MOCK_REGIONS.flatMap((region) => {
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
  });
  return {
    status: 'DATA DUMMY · fallback karena backend tidak tersedia',
    source: 'frontend/src/data/mockData.ts',
    observationPeriod: ['2005-04-01', '2026-01-01'],
    predictionPeriod: ['2024-01-01', '2025-12-03'],
    grid,
    regions: MOCK_REGIONS,
  };
}
