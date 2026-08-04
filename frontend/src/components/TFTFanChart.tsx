import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  TooltipProps,
} from 'recharts';
import { RegionPrediction } from '../types';

export interface FanChartDataPoint {
  month: string;
  isForecast: boolean;
  actual?: number;
  q10?: number;
  q50?: number;
  q90?: number;
  // Area stack helpers for fan bands (q10 to q50, q50 to q90)
  q10_q50_diff?: number;
  q50_q90_diff?: number;
}

interface TFTFanChartProps {
  region: RegionPrediction;
  selectedHorizon?: number; // 1, 3, 6, or 12
  onHorizonChange?: (horizon: number) => void;
  className?: string;
}

// Generate data combining 6M historical trajectory + +1M..+12M quantile forecast
export function generateFanChartData(region: RegionPrediction, horizonsCount = 12): FanChartDataPoint[] {
  const data: FanChartDataPoint[] = [];

  // Historical points
  region.historicalSpei.forEach((h) => {
    data.push({
      month: h.month,
      isForecast: false,
      actual: h.actual,
    });
  });

  const lastHist = region.historicalSpei[region.historicalSpei.length - 1];
  const lastActual = lastHist ? lastHist.actual : region.speiCurrent;

  // Seamless connection at forecast start (Horizon +0)
  // Stack diffs = 0 at start point
  data[data.length - 1] = {
    ...data[data.length - 1],
    q10: lastActual,
    q50: lastActual,
    q90: lastActual,
    q10_q50_diff: 0,
    q50_q90_diff: 0,
  };

  // Generate +1M to +12M horizons
  const forecastMonths = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan+1', 'Feb+1', 'Mar+1', 'Apr+1', 'May+1', 'Jun+1'];
  
  const targetQ10 = region.speiForecast.q10;
  const targetQ50 = region.speiForecast.q50;
  const targetQ90 = region.speiForecast.q90;

  for (let i = 1; i <= horizonsCount; i++) {
    const monthName = forecastMonths[(i - 1) % forecastMonths.length];

    // Fan uncertainty widens as forecast horizon extends
    const factor = Math.sqrt(i / 3); 
    const q50Val = Number((lastActual + (targetQ50 - lastActual) * (i / 3)).toFixed(2));
    const q10Val = Number((q50Val + (targetQ10 - targetQ50) * factor).toFixed(2));
    const q90Val = Number((q50Val + (targetQ90 - targetQ50) * factor).toFixed(2));

    data.push({
      month: `${monthName} (+${i}M)`,
      isForecast: true,
      q10: q10Val,
      q50: q50Val,
      q90: q90Val,
      q10_q50_diff: Number((q50Val - q10Val).toFixed(2)),
      q50_q90_diff: Number((q90Val - q50Val).toFixed(2)),
    });
  }

  return data;
}

// Custom Chrome Tooltip with SPEI severity breakdown & quantile spread
const CustomTooltip: React.FC<TooltipProps<number, string>> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  const dataPoint = payload[0]?.payload as FanChartDataPoint;
  if (!dataPoint) return null;

  const getSpeiCategory = (val?: number) => {
    if (val === undefined) return { label: 'N/A', color: 'text-[var(--muted)]' };
    if (val <= -2.0) return { label: 'Kekeringan ekstrem', color: 'text-[var(--red)]' };
    if (val <= -1.5) return { label: 'Kekeringan parah', color: 'text-[#b45b2c]' };
    if (val <= -0.5) return { label: 'Kekeringan sedang', color: 'text-[var(--amber)]' };
    if (val < 0.5) return { label: 'Normal', color: 'text-[var(--green)]' };
    return { label: 'Lebih basah dari normal', color: 'text-[var(--blue)]' };
  };

  const speiStatus = getSpeiCategory(dataPoint.isForecast ? dataPoint.q50 : dataPoint.actual);

  return (
    <div className="bg-[var(--surface)]/95 border border-[var(--line)] p-3 rounded-md shadow-lg backdrop-blur font-mono text-xs max-w-xs space-y-2">
      <div className="flex items-center justify-between border-b border-[var(--line)] pb-1.5">
        <span className="font-bold text-[var(--ink)]">{label}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded border ${
          dataPoint.isForecast 
            ? 'bg-[var(--soft-green)] text-[var(--green)] border-[var(--green)]' 
            : 'bg-[#f0f2ee] text-[var(--muted)] border-[var(--line)]'
        }`}>
          {dataPoint.isForecast ? 'HASIL EVALUASI MODEL' : 'OBSERVASI'}
        </span>
      </div>

      <div className="space-y-1">
        <div className="text-[11px] font-semibold flex justify-between">
          <span className="text-[var(--muted)]">Kondisi:</span>
          <span className={speiStatus.color}>{speiStatus.label}</span>
        </div>

        {!dataPoint.isForecast && dataPoint.actual !== undefined && (
          <div className="flex justify-between text-[var(--ink)]">
            <span className="text-[var(--muted)]">SPEI teramati:</span>
            <span className="font-bold text-[var(--blue)]">{dataPoint.actual.toFixed(2)}</span>
          </div>
        )}

        {dataPoint.isForecast && (
          <>
            <div className="flex justify-between text-[var(--green)]">
              <span>q0.90 (batas lebih basah):</span>
              <span className="font-bold">{dataPoint.q90?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-[var(--blue)] font-semibold bg-[var(--blue-soft)] px-1 py-0.5 rounded">
              <span>q0.50 (nilai tengah):</span>
              <span className="font-bold">{dataPoint.q50?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-[var(--red)]">
              <span>q0.10 (batas lebih kering):</span>
              <span className="font-bold">{dataPoint.q10?.toFixed(2)}</span>
            </div>
            {dataPoint.q90 !== undefined && dataPoint.q10 !== undefined && (
              <div className="flex justify-between text-[10px] text-[var(--muted)] pt-1 border-t border-[var(--line)]">
                <span>Rentang kemungkinan 80%:</span>
                <span>{(dataPoint.q90 - dataPoint.q10).toFixed(2)} ΔSPEI</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export const TFTFanChart: React.FC<TFTFanChartProps> = ({
  region,
  selectedHorizon = 3,
  onHorizonChange,
  className = '',
}) => {
  const [showUncertaintyBands, setShowUncertaintyBands] = useState(true);
  const chartData = generateFanChartData(region, 12);

  const activeHorizonPoint = chartData.find((d) => d.month.includes(`(+${selectedHorizon}M)`));

  return (
    <div className={`bg-[#f3f6f2] border-[var(--line)] rounded-lg p-5 flex flex-col space-y-4 ${className}`}>
      
      {/* Header & Controls Chrome */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[var(--green)] animate-pulse"></span>
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-gray-200">
              Prediksi SPEI dengan rentang kemungkinan ({region.regencyName})
            </h3>
          </div>
          <p className="text-[11px] font-mono text-gray-400 mt-0.5">
            Garis putus-putus = nilai tengah; pita = rentang kemungkinan 80%.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {/* Horizon Selection Buttons */}
          <div className="flex items-center bg-gray-900 border-[var(--line)] rounded p-0.5">
            {[1, 3, 6, 12].map((h) => (
              <button
                key={h}
                onClick={() => onHorizonChange?.(h)}
                className={`px-2 py-0.5 text-[11px] font-mono rounded transition-colors ${
                  selectedHorizon === h
                    ? 'bg-[var(--blue-soft)] text-[var(--blue)] border border-[var(--blue)] font-semibold'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                +{h}M
              </button>
            ))}
          </div>

          {/* Toggle Band Visibility */}
          <button
            onClick={() => setShowUncertaintyBands(!showUncertaintyBands)}
            className={`px-2 py-1 text-[10px] font-mono rounded border transition-colors ${
              showUncertaintyBands
                ? 'bg-gray-800 text-gray-200 border-gray-700'
                : 'bg-gray-900 text-gray-500 var(--line) line-through'
            }`}
          >
            PITA
          </button>
        </div>
      </div>

      {/* Main Fan-Chart Visual Canvas */}
      <div className="h-64 w-full bg-[var(--surface)] border-[var(--line)] rounded p-2 relative">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
            <defs>
              {/* Outer 80% CI Shading (q10 to q90 spread) */}
              <linearGradient id="fanOuterBand" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#56708a" stopOpacity={0.05} />
                <stop offset="100%" stopColor="#56708a" stopOpacity={0.25} />
              </linearGradient>
              {/* Inner 50% / Median Shading */}
              <linearGradient id="fanInnerBand" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#4c7a78" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#4c7a78" stopOpacity={0.4} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#d4dbd3" vertical={false} />
            <XAxis
              dataKey="month"
              stroke="#6b7280"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: '#374151' }}
            />
            <YAxis
              stroke="#6b7280"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: '#374151' }}
              domain={[-3.0, 1.5]}
              ticks={[-2.5, -2.0, -1.5, -0.5, 0, 0.5, 1.0]}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Threshold Reference Lines for Drought Classification */}
            <ReferenceLine y={-2.0} stroke="#a33d2e" strokeDasharray="4 4" label={{ value: 'EXTREME (-2.0)', fill: '#a33d2e', fontSize: 9, position: 'right' }} />
            <ReferenceLine y={-1.5} stroke="#b45b2c" strokeDasharray="4 4" label={{ value: 'SEVERE (-1.5)', fill: '#b45b2c', fontSize: 9, position: 'right' }} />
            <ReferenceLine y={-0.5} stroke="#9a5b00" strokeDasharray="4 4" label={{ value: 'MODERATE (-0.5)', fill: '#9a5b00', fontSize: 9, position: 'right' }} />
            <ReferenceLine y={0} stroke="#9aa8a0" strokeDasharray="2 2" />

            {/* Selected Horizon Highlight Line */}
            {activeHorizonPoint && (
              <ReferenceLine
                x={activeHorizonPoint.month}
                stroke="#56708a"
                strokeWidth={1.5}
                strokeDasharray="2 2"
              />
            )}

            {/* Quantile Fan Band 1: Base stack to q10 (Transparent) */}
            {showUncertaintyBands && (
              <Area
                type="monotone"
                dataKey="q10"
                stackId="fan"
                stroke="none"
                fill="none"
                isAnimationActive={false}
              />
            )}

            {/* Quantile Fan Band 2: q10 to q50 (Lower Fan Area) */}
            {showUncertaintyBands && (
              <Area
                type="monotone"
                dataKey="q10_q50_diff"
                stackId="fan"
                stroke="none"
                fill="url(#fanOuterBand)"
                name="q10-q50 Band"
                isAnimationActive={false}
              />
            )}

            {/* Quantile Fan Band 3: q50 to q90 (Upper Fan Area) */}
            {showUncertaintyBands && (
              <Area
                type="monotone"
                dataKey="q50_q90_diff"
                stackId="fan"
                stroke="none"
                fill="url(#fanInnerBand)"
                name="q50-q90 Band"
                isAnimationActive={false}
              />
            )}

            {/* Historical Actual Line */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#56708a"
              strokeWidth={2}
              dot={{ r: 3, fill: '#56708a' }}
              name="Historical SPEI"
            />

            {/* Quantile q50 Median Forecast Line */}
            <Line
              type="monotone"
              dataKey="q50"
              stroke="#9a5b00"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={false}
              name="TFT Median (q0.50)"
            />

            {/* Quantile q10 Boundary Line */}
            {showUncertaintyBands && (
              <Line
                type="monotone"
                dataKey="q10"
                stroke="#a33d2e"
                strokeWidth={1}
                strokeDasharray="2 2"
                dot={false}
                name="Worst Case (q0.10)"
              />
            )}

            {/* Quantile q90 Boundary Line */}
            {showUncertaintyBands && (
              <Line
                type="monotone"
                dataKey="q90"
                stroke="#0b6f68"
                strokeWidth={1}
                strokeDasharray="2 2"
                dot={false}
                name="Best Case (q0.90)"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend & Metric Summary */}
      <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-gray-400 gap-2 bg-[#ffffff] p-2.5 rounded border-[var(--line)]">
        <div className="flex items-center space-x-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-[var(--blue)] inline-block"></span>
            Historical SPEI
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-amber-500 border-b border-dashed border-amber-500 inline-block"></span>
            q50 Median
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 bg-[var(--blue)]/30 border border-[var(--blue)]/50 rounded-sm inline-block"></span>
            q10 - q90 Fan (80% CI)
          </span>
        </div>

        {activeHorizonPoint && (
          <div className="flex items-center space-x-2 text-[var(--ink)]">
            <span>Hasil +{selectedHorizon} bulan:</span>
            <span className="text-[var(--red)] font-bold">q10: {activeHorizonPoint.q10}</span>
            <span>|</span>
            <span className="text-[var(--blue)] font-bold">q50: {activeHorizonPoint.q50}</span>
            <span>|</span>
            <span className="text-[var(--green)] font-bold">q90: {activeHorizonPoint.q90}</span>
          </div>
        )}
      </div>
    </div>
  );
};
