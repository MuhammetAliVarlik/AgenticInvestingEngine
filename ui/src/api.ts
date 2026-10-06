import type {
  DatasetUploadResponse,
  DocumentUploadResponse,
  HistoricalPrediction,
  Instrument,
  StreamEvent,
  UsageStats,
  UserSession,
} from './types';

/**
 * Mock mode is a build-time switch (`VITE_MOCK=1`) for design previews only.
 * The real client never falls back to mock data: if the backend fails, the
 * user sees the error, never invented results.
 */
export const MOCK_MODE = import.meta.env.VITE_MOCK === '1';

const mock = () => import('./mock');

/** Header the gateway requires on every state-changing request (CSRF defence). */
const CSRF_HEADER = { 'X-Requested-With': 'investing-engine' };

export class ApiError extends Error {
  status: number;
  detail: string;
  login_url?: string;
  logout_url?: string;
  method?: 'code' | 'redirect' | 'none';

  constructor(
    status: number,
    detail: string,
    extra: { login_url?: string; logout_url?: string; method?: ApiError['method'] } = {},
  ) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.login_url = extra.login_url;
    this.logout_url = extra.logout_url;
    this.method = extra.method;
  }
}

async function failure(res: Response, fallback: string): Promise<ApiError> {
  const data = await res.json().catch(() => ({}));
  const detail = typeof data.detail === 'string' ? data.detail : fallback;
  return new ApiError(res.status, detail, {
    login_url: data.login_url,
    logout_url: data.logout_url,
    method: data.method,
  });
}

async function getJson<T>(path: string, fallback: string): Promise<T> {
  const res = await fetch(`/api/${path}`, { credentials: 'same-origin' });
  if (!res.ok) throw await failure(res, fallback);
  return (await res.json()) as T;
}

export async function fetchCurrentUser(): Promise<UserSession> {
  if (MOCK_MODE) return (await mock()).getMockUser();
  return getJson<UserSession>('me', 'Could not read the session');
}

export async function redeemAccessCode(code: string): Promise<UserSession> {
  const res = await fetch('/auth/code', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { ...CSRF_HEADER, 'Content-Type': 'application/json' },
    body: JSON.stringify({ code }),
  });
  if (!res.ok) throw await failure(res, 'This access code was not accepted');
  return (await res.json()) as UserSession;
}

export async function fetchInstruments(): Promise<Instrument[]> {
  if (MOCK_MODE) return (await mock()).getMockInstruments();
  return getJson<Instrument[]>('instruments', 'Could not load the instruments');
}

export async function fetchUsage(): Promise<UsageStats> {
  if (MOCK_MODE) return (await mock()).getMockUsage();
  return getJson<UsageStats>('usage', 'Could not load the usage limits');
}

export async function fetchHistory(symbol: string): Promise<HistoricalPrediction[]> {
  if (MOCK_MODE) return (await mock()).getMockHistory(symbol);
  return getJson<HistoricalPrediction[]>(
    `history/${encodeURIComponent(symbol)}`,
    `Could not load the history of ${symbol}`,
  );
}

async function postFile<T>(path: string, symbol: string, file: File, fallback: string): Promise<T> {
  const form = new FormData();
  form.append('symbol', symbol);
  form.append('file', file, file.name);
  const res = await fetch(`/api/${path}`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: CSRF_HEADER,
    body: form,
  });
  if (!res.ok) throw await failure(res, fallback);
  return (await res.json()) as T;
}

export async function uploadDataset(symbol: string, file: File): Promise<DatasetUploadResponse> {
  if (MOCK_MODE) return (await mock()).uploadMockDataset(symbol, file);
  return postFile('datasets', symbol, file, 'The price file was not accepted');
}

export async function uploadDocument(symbol: string, file: File): Promise<DocumentUploadResponse> {
  if (MOCK_MODE) return (await mock()).uploadMockDocument(symbol, file);
  return postFile('documents', symbol, file, 'The document was not accepted');
}

export async function streamAnalysis(
  payload: {
    symbols: string[];
    datasets: Record<string, string>;
    documents: Record<string, string>;
  },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (MOCK_MODE) {
    const hasDisclosures = Object.keys(payload.documents).length > 0;
    return (await mock()).runMockAnalysisStream(payload.symbols, hasDisclosures, onEvent, signal);
  }

  const res = await fetch('/api/analyses/stream', {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...CSRF_HEADER,
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) {
    throw await failure(res, `The analysis could not start (HTTP ${res.status})`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      }
      boundary = buffer.indexOf('\n\n');
    }
  }
}

export function getPdfReportUrl(analysisId: string): string {
  return `/api/analyses/${encodeURIComponent(analysisId)}/report.pdf`;
}
