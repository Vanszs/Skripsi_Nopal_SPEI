import { useState, useEffect, useRef, useCallback } from 'react';

export interface WeatherData {
  temp_c: number;
  humidity_pct: number;
  precip_mm: number;
  et0_mm: number;
}

export interface InferenceState {
  model_status: string;
  latest_spei_p50: number;
  drought_risk: string;
}

export interface StreamPayload {
  type: string;
  city_id: string;
  timestamp: string;
  step: number;
  weather: WeatherData;
  inference_state: InferenceState;
}

interface UseDroughtStreamOptions {
  cityId?: string;
  mode?: 'sse' | 'websocket';
  sseUrl?: string;
  wsUrl?: string;
  enabled?: boolean;
}

export function useDroughtStream({
  cityId = 'surabaya',
  mode = 'sse',
  sseUrl = '/api/v1/stream/weather',
  wsUrl = '/ws/monitoring',
  enabled = true,
}: UseDroughtStreamOptions = {}) {
  const [data, setData] = useState<StreamPayload | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Connection setup
  useEffect(() => {
    if (!enabled) return;

    setError(null);

    if (mode === 'sse') {
      const url = `${sseUrl}?city_id=${encodeURIComponent(cityId)}`;
      const eventSource = new EventSource(url);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setIsConnected(true);
      };

      eventSource.addEventListener('weather_update', (event) => {
        try {
          const parsed = JSON.parse(event.data) as StreamPayload;
          setData(parsed);
        } catch (e) {
          console.error('Failed to parse SSE payload', e);
        }
      });

      eventSource.onerror = () => {
        setIsConnected(false);
        setError('SSE connection error');
        eventSource.close();
      };

      return () => {
        eventSource.close();
        setIsConnected(false);
      };
    } else if (mode === 'websocket') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const fullWsUrl = wsUrl.startsWith('ws') ? wsUrl : `${protocol}//${window.location.host}${wsUrl}`;
      const ws = new WebSocket(fullWsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        ws.send(JSON.stringify({ action: 'subscribe', city_id: cityId }));
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.weather) {
            setData(parsed as StreamPayload);
          }
        } catch (e) {
          console.error('Failed to parse WS payload', e);
        }
      };

      ws.onerror = () => {
        setIsConnected(false);
        setError('WebSocket error');
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      return () => {
        ws.close();
        setIsConnected(false);
      };
    }
  }, [cityId, mode, sseUrl, wsUrl, enabled]);

  const sendWsMessage = useCallback((msg: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  }, []);

  return { data, isConnected, error, sendWsMessage };
}
