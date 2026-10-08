export interface AccessCodeInfo {
  id: string;
  expires_at: string;
  quota: number;
  used: number;
}

export interface UserSession {
  user: string;
  provider: 'none' | 'easyauth' | 'oidc' | 'accesscode';
  logout_url: string | null;
  features?: { documents: boolean };
  code?: AccessCodeInfo;
}

export interface AuthError {
  detail: string;
  login_url?: string;
  logout_url?: string;
  method?: 'code' | 'redirect' | 'none';
  /** 401 sign-in, 403 refused, other values: the service is not available. */
  status: number;
}

export interface Instrument {
  symbol: string;
  name: string;
  kind: 'index' | 'equity';
  public_prices: boolean;
}

export interface UsageStats {
  analyses: number;
  analyses_limit: number;
  tokens: number;
  tokens_limit: number;
}

export interface HistoricalPrediction {
  timestamp: string;
  signal: string | null;
  risk_score: number | null;
  price_at_prediction: number | null;
  /** True when the analysis used the viewer's own files and only they can see it. */
  private?: boolean;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  symbol: string;
  rows: number;
  start: string;
  end: string;
  columns: string[];
}

export interface DocumentUploadResponse {
  document_id: string;
  symbol: string;
  pages: number;
  ocr_pages: number;
  injection_flags?: string[];
}

export type AgentName =
  | 'supervisor'
  | 'technical_analyst'
  | 'news_analyst'
  | 'macro_analyst'
  | 'disclosure_analyst';

export type ToolName =
  | 'technical_snapshot'
  | 'news_headlines'
  | 'macro_snapshot'
  | 'disclosure_document'
  | 'prediction_history';

export interface StreamStatusEvent {
  type: 'status';
  text:
    | 'Consulting technical_analyst'
    | 'Consulting news_analyst'
    | 'Consulting macro_analyst'
    | 'Consulting disclosure_analyst'
    | 'Writing the report'
    | string;
}

export interface StreamToolCallEvent {
  type: 'tool_call';
  agent: string;
  tool: string;
}

export interface StreamToolResultEvent {
  type: 'tool_result';
  agent: string;
  tool: string;
}

export interface StreamTokenEvent {
  type: 'token';
  agent: string;
  text: string;
}

export interface StreamErrorEvent {
  type: 'error';
  message: string;
}

export interface SourceAttribution {
  name: string;
  attribution: string;
}

export interface AgentUsage {
  input_tokens: number;
  output_tokens: number;
  calls: number;
}

export interface QualityChecks {
  grounding_score?: number | null; // 0-1
  checked_figures?: number;
  ungrounded_figures?: string[];
  unexpected_symbols?: string[];
  /** Required specialists the supervisor did not consult in this run. */
  missing_specialists?: string[];
}

export interface TechnicalIndicatorData {
  price?: number | string | null;
  rsi?: number | string | null;
  predicted_next_rsi?: number | string | null;
  ema34?: number | string | null;
  ema89?: number | string | null;
  price_above_ema34?: boolean | string | null;
  macd?: number | string | null;
  bb_pct?: number | string | null;
  channel_position?: string | null;
  momentum_divergence?: string | null;
  signal?: 'bullish' | 'bearish' | 'neutral' | string | null;
  as_of?: string | null;
  source?: string | null;
  [key: string]: string | number | boolean | null | undefined;
}

export interface StreamFinalEvent {
  type: 'final';
  report: string;
  technical: Record<string, Record<string, string | number | boolean | null>>;
  risk: Record<string, number>; // 0-10
  sources: SourceAttribution[];
  timings: {
    total_seconds?: number;
  };
  usage: Record<string, AgentUsage>;
  checks: QualityChecks;
  injection_flags: string[];
  cached: boolean;
  analysis_id: string | null;
}

export type StreamEvent =
  | StreamStatusEvent
  | StreamToolCallEvent
  | StreamToolResultEvent
  | StreamTokenEvent
  | StreamErrorEvent
  | StreamFinalEvent;

export type PipelineStageId =
  | 'verify_inputs'
  | 'plan_analysis'
  | 'technical_analysis'
  | 'news_risk'
  | 'macro_backdrop'
  | 'disclosures'
  | 'write_report'
  | 'quality_checks';

export type StageStatus = 'waiting' | 'running' | 'done' | 'not_needed' | 'failed';

export interface PipelineStage {
  id: PipelineStageId;
  name: string;
  agent?: AgentName;
  status: StageStatus;
  toolsUsed: string[];
  startedAt?: number;
  completedAt?: number;
}

export interface ActivityLogItem {
  id: string;
  timestamp: string; // HH:mm:ss
  text: string;
  type: 'status' | 'tool' | 'token' | 'warning' | 'error' | 'success';
}

export interface UploadedDataState {
  dataset?: DatasetUploadResponse;
  document?: DocumentUploadResponse;
  datasetUploading?: boolean;
  documentUploading?: boolean;
  datasetError?: string;
  documentError?: string;
  rawFile?: File;
}
