export type DroughtSeverity = 'NORMAL' | 'MODERATE' | 'SEVERE' | 'EXTREME';

export interface QuantileForecast {
  q10: number; // Worst case scenario (10th percentile)
  q50: number; // Median prediction (50th percentile)
  q90: number; // Best case scenario (90th percentile)
}

export interface RegionPrediction {
  id: string;
  regencyName: string;
  province: string;
  speiCurrent: number;
  speiForecast: QuantileForecast;
  severity: DroughtSeverity;
  coordinates: [number, number]; // [lat, lng]
  historicalSpei: { month: string; actual: number; predicted: number }[];
}

export interface ModelMetrics {
  quantileLoss: number;
  rmse: number;
  mae: number;
  skillScore: number;
  dropoutRate: number;
  multiSeedVerified: boolean;
  commitHash: string;
}
