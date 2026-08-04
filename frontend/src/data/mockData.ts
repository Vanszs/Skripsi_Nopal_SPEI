import { RegionPrediction, ModelMetrics } from '../types';

export const SYSTEM_METRICS: ModelMetrics = {
  quantileLoss: 0.142,
  rmse: 0.285,
  mae: 0.210,
  skillScore: 0.865,
  dropoutRate: 0.40,
  multiSeedVerified: true,
  commitHash: '87eae914ad6e296bfc98b85939be7641139248b4'
};

export const MOCK_REGIONS: RegionPrediction[] = [
  {
    id: 'Bojonegoro',
    regencyName: 'Kab. Bojonegoro',
    province: 'Jawa Timur',
    speiCurrent: -1.62,
    speiForecast: { q10: -2.45, q50: -1.98, q90: -1.35 },
    severity: 'SEVERE',
    coordinates: [-7.155, 111.880],
    historicalSpei: [
      { month: 'Jan', actual: -0.20, predicted: -0.25 },
      { month: 'Feb', actual: -0.45, predicted: -0.40 },
      { month: 'Mar', actual: -0.80, predicted: -0.75 },
      { month: 'Apr', actual: -1.10, predicted: -1.15 },
      { month: 'May', actual: -1.40, predicted: -1.38 },
      { month: 'Jun', actual: -1.62, predicted: -1.60 }
    ]
  },
  {
    id: 'Lamongan',
    regencyName: 'Kab. Lamongan',
    province: 'Jawa Timur',
    speiCurrent: -0.85,
    speiForecast: { q10: -1.65, q50: -1.20, q90: -0.62 },
    severity: 'MODERATE',
    coordinates: [-7.128, 112.316],
    historicalSpei: [
      { month: 'Jan', actual: 0.30, predicted: 0.25 },
      { month: 'Feb', actual: 0.10, predicted: 0.05 },
      { month: 'Mar', actual: -0.25, predicted: -0.20 },
      { month: 'Apr', actual: -0.50, predicted: -0.48 },
      { month: 'May', actual: -0.70, predicted: -0.68 },
      { month: 'Jun', actual: -0.85, predicted: -0.82 }
    ]
  },
  {
    id: 'Nganjuk',
    regencyName: 'Kab. Nganjuk',
    province: 'Jawa Timur',
    speiCurrent: -0.30,
    speiForecast: { q10: -1.15, q50: -0.65, q90: -0.10 },
    severity: 'NORMAL',
    coordinates: [-7.604, 111.905],
    historicalSpei: [
      { month: 'Jan', actual: 0.60, predicted: 0.55 },
      { month: 'Feb', actual: 0.40, predicted: 0.35 },
      { month: 'Mar', actual: 0.15, predicted: 0.10 },
      { month: 'Apr', actual: -0.05, predicted: -0.02 },
      { month: 'May', actual: -0.20, predicted: -0.18 },
      { month: 'Jun', actual: -0.30, predicted: -0.28 }
    ]
  },
  {
    id: 'Ngawi',
    regencyName: 'Kab. Ngawi',
    province: 'Jawa Timur',
    speiCurrent: -2.15,
    speiForecast: { q10: -2.90, q50: -2.45, q90: -1.90 },
    severity: 'EXTREME',
    coordinates: [-7.403, 111.445],
    historicalSpei: [
      { month: 'Jan', actual: -0.80, predicted: -0.75 },
      { month: 'Feb', actual: -1.10, predicted: -1.05 },
      { month: 'Mar', actual: -1.45, predicted: -1.40 },
      { month: 'Apr', actual: -1.70, predicted: -1.68 },
      { month: 'May', actual: -1.95, predicted: -1.90 },
      { month: 'Jun', actual: -2.15, predicted: -2.10 }
    ]
  },
  {
    id: 'Tuban',
    regencyName: 'Kab. Tuban',
    province: 'Jawa Timur',
    speiCurrent: -1.75,
    speiForecast: { q10: -2.55, q50: -2.05, q90: -1.48 },
    severity: 'SEVERE',
    coordinates: [-6.895, 112.045],
    historicalSpei: [
      { month: 'Jan', actual: -0.40, predicted: -0.35 },
      { month: 'Feb', actual: -0.70, predicted: -0.65 },
      { month: 'Mar', actual: -1.05, predicted: -1.00 },
      { month: 'Apr', actual: -1.35, predicted: -1.30 },
      { month: 'May', actual: -1.58, predicted: -1.55 },
      { month: 'Jun', actual: -1.75, predicted: -1.72 }
    ]
  }
];
