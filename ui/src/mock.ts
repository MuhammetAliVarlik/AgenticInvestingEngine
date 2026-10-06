import {
  UserSession,
  Instrument,
  UsageStats,
  HistoricalPrediction,
  DatasetUploadResponse,
  DocumentUploadResponse,
  StreamEvent,
  StreamFinalEvent
} from './types';

// 32 Instruments: XU100 Index + BIST 30 Equities + ARCLK
export const MOCK_INSTRUMENTS: Instrument[] = [
  { symbol: 'XU100', name: 'BIST 100 Endeksi', kind: 'index', public_prices: true },
  { symbol: 'AKBNK', name: 'Akbank T.A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ALARK', name: 'Alarko Holding A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ARCLK', name: 'Arçelik A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ASELS', name: 'Aselsan Elektronik Sanayi ve Ticaret A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ASTOR', name: 'Astor Enerji A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'BIMAS', name: 'BİM Birleşik Mağazalar A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'BRSAN', name: 'Borusan Boru Sanayi ve Ticaret A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'EKGYO', name: 'Emlak Konut Gayrimenkul Yatırım Ortaklığı A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ENKAI', name: 'Enka İnşaat ve Sanayi A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'EREGL', name: 'Ereğli Demir ve Çelik Fabrikaları T.A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'FROTO', name: 'Ford Otomotiv Sanayi A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'GARAN', name: 'Türkiye Garanti Bankası A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'GUBRF', name: 'Gübre Fabrikaları T.A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'HEKTS', name: 'Hektaş Ticaret T.A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'ISCTR', name: 'Türkiye İş Bankası A.Ş. (C)', kind: 'equity', public_prices: false },
  { symbol: 'KCHOL', name: 'Koç Holding A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'KONTR', name: 'Kontrolmatik Teknoloji Enerji ve Mühendislik A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'KOZAL', name: 'Koza Altın İşletmeleri A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'KRDMD', name: 'Kardemir Karabük Demir Çelik Sanayi (D)', kind: 'equity', public_prices: false },
  { symbol: 'MGROS', name: 'Migros Ticaret A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'OYAKC', name: 'OYAK Çimento Fabrikaları A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'PETKM', name: 'Petkim Petrokimya Holding A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'PGSUS', name: 'Pegasus Hava Taşımacılığı A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'SAHOL', name: 'Hacı Ömer Sabancı Holding A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'SASA', name: 'SASA Polyester Sanayi A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'SISE', name: 'Türkiye Şişe ve Cam Fabrikaları A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'TCELL', name: 'Turkcell İletişim Hizmetleri A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'THYAO', name: 'Türk Hava Yolları A.O.', kind: 'equity', public_prices: false },
  { symbol: 'TOASO', name: 'Tofaş Türk Otomobil Fabrikası A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'TUPRS', name: 'Tüpraş - Türkiye Petrol Rafinerileri A.Ş.', kind: 'equity', public_prices: false },
  { symbol: 'YKBNK', name: 'Yapı ve Kredi Bankası A.Ş.', kind: 'equity', public_prices: false }
];

export const MOCK_USER: UserSession = {
  user: 'Mock data (preview)',
  provider: 'none',
  logout_url: null,
  features: { documents: true }
};

let currentUsage: UsageStats = {
  analyses: 2,
  analyses_limit: 5,
  tokens: 42180,
  tokens_limit: 150000
};

export function getMockUser(): Promise<UserSession> {
  return new Promise((resolve) => setTimeout(() => resolve({ ...MOCK_USER }), 150));
}

export function getMockUsage(): Promise<UsageStats> {
  return new Promise((resolve) => setTimeout(() => resolve({ ...currentUsage }), 150));
}

export function incrementMockUsage(tokensUsed: number = 7400) {
  currentUsage = {
    ...currentUsage,
    analyses: Math.min(currentUsage.analyses_limit, currentUsage.analyses + 1),
    tokens: currentUsage.tokens + tokensUsed
  };
}

export function getMockInstruments(): Promise<Instrument[]> {
  return new Promise((resolve) => setTimeout(() => resolve([...MOCK_INSTRUMENTS]), 180));
}

export function getMockHistory(symbol: string): Promise<HistoricalPrediction[]> {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Deterministic baseline dates and figures for the stock
      const basePrice = symbol === 'THYAO' ? 312.5 : symbol === 'ASELS' ? 64.2 : symbol === 'XU100' ? 9840.0 : symbol === 'KCHOL' ? 195.4 : 120.0;
      const history: HistoricalPrediction[] = [
        {
          timestamp: '2026-08-14T09:30:00Z',
          signal: 'bullish',
          risk_score: 2.4,
          price_at_prediction: Number((basePrice * 0.91).toFixed(2))
        },
        {
          timestamp: '2026-08-28T09:30:00Z',
          signal: 'bullish',
          risk_score: 2.8,
          price_at_prediction: Number((basePrice * 0.94).toFixed(2))
        },
        {
          timestamp: '2026-09-08T09:30:00Z',
          signal: 'neutral',
          risk_score: 4.5,
          price_at_prediction: Number((basePrice * 0.97).toFixed(2))
        },
        {
          timestamp: '2026-09-18T09:30:00Z',
          signal: 'bullish',
          risk_score: 3.1,
          price_at_prediction: Number((basePrice * 0.985).toFixed(2))
        },
        {
          timestamp: '2026-09-25T09:30:00Z',
          signal: 'bearish',
          risk_score: 6.2,
          price_at_prediction: Number((basePrice * 1.02).toFixed(2))
        },
        {
          timestamp: '2026-10-02T09:30:00Z',
          signal: 'bullish',
          risk_score: 3.4,
          price_at_prediction: Number(basePrice.toFixed(2))
        }
      ];
      resolve(history);
    }, 200);
  });
}

export function uploadMockDataset(symbol: string, _file: File): Promise<DatasetUploadResponse> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        dataset_id: `ds_${symbol.toLowerCase()}_${Date.now().toString(36)}`,
        symbol,
        rows: 502,
        start: '2024-10-03',
        end: '2026-10-02',
        columns: ['Tarih', 'Kapanis', 'Aof', 'En Yuksek', 'En Dusuk', 'Hacim']
      });
    }, 600);
  });
}

export function uploadMockDocument(symbol: string, file: File): Promise<DocumentUploadResponse> {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Occasional non-critical injection flag detection on disclosures for realism
      const isSuspect = file.name.toLowerCase().includes('prompt') || file.name.toLowerCase().includes('ignore');
      resolve({
        document_id: `doc_${symbol.toLowerCase()}_${Date.now().toString(36)}`,
        symbol,
        pages: 18,
        ocr_pages: 4,
        injection_flags: isSuspect ? ['Indirect instruction override stripped in Section 4'] : []
      });
    }, 700);
  });
}

// Generate rich mock technical indicator payload
function generateMockTechnical(symbol: string): Record<string, string | number | boolean | null> {
  const isThyao = symbol === 'THYAO';
  const isAsels = symbol === 'ASELS';
  const isIndex = symbol === 'XU100';

  if (isThyao) {
    return {
      price: 326.5,
      rsi: 58.4,
      predicted_next_rsi: 61.2,
      ema34: 312.2,
      ema89: 294.8,
      price_above_ema34: true,
      macd: 4.82,
      bb_pct: 72.4,
      channel_position: 'Upper quartile',
      momentum_divergence: 'None (Confirmed trend)',
      signal: 'bullish',
      as_of: '2026-10-06 17:45 TRT',
      source: 'İş Yatırım Tarihsel Fiyat Verisi (502 barlar)'
    };
  }

  if (isAsels) {
    return {
      price: 66.8,
      rsi: 64.1,
      predicted_next_rsi: 67.5,
      ema34: 62.4,
      ema89: 58.1,
      price_above_ema34: true,
      macd: 1.45,
      bb_pct: 78.1,
      channel_position: 'Upper band test',
      momentum_divergence: 'Bullish continuation',
      signal: 'bullish',
      as_of: '2026-10-06 17:45 TRT',
      source: 'İş Yatırım Tarihsel Fiyat Verisi (502 barlar)'
    };
  }

  if (isIndex) {
    return {
      price: 9942.3,
      rsi: 53.2,
      predicted_next_rsi: 54.8,
      ema34: 9780.5,
      ema89: 9610.2,
      price_above_ema34: true,
      macd: 28.4,
      bb_pct: 59.3,
      channel_position: 'Mid-channel',
      momentum_divergence: 'Mild compression',
      signal: 'neutral',
      as_of: '2026-10-06 17:45 TRT',
      source: 'BIST Endeks Canlı / Kamu Kaynakları'
    };
  }

  return {
    price: 184.2,
    rsi: 48.9,
    predicted_next_rsi: 50.1,
    ema34: 182.1,
    ema89: 179.4,
    price_above_ema34: true,
    macd: 0.92,
    bb_pct: 51.6,
    channel_position: 'Median',
    momentum_divergence: 'Neutral',
    signal: 'neutral',
    as_of: '2026-10-06 17:45 TRT',
    source: 'İş Yatırım Tarihsel Fiyat Verisi (502 barlar)'
  };
}

export function runMockAnalysisStream(
  symbols: string[],
  hasDisclosures: boolean,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  return new Promise((resolve, reject) => {
    const symbolsList = symbols.join(', ');
    const isAborted = () => signal?.aborted;

    const reportTokens = [
      `# Executive Investment Synthesis: Borsa İstanbul\n\n`,
      `**Coverage Focus:** ${symbolsList} | **Date:** October 6, 2026 | **Framework:** Multi-Agent BIST Architecture\n\n`,
      `---\n\n`,
      `## 1. Macro Backdrop & CBRT Policy Transmission\n\n`,
      `The Central Bank of the Republic of Türkiye (CBRT) continues its orthodox policy anchoring with the 1-week repo rate held at 45.0%. Disinflation dynamics are steadily compressing headline CPI toward target bands, supporting foreign portfolio inflows into Turkish equities while stabilizing the USD/TRY spread volatility.\n\n`,
      `- **Sovereign CDS (5Y):** 254 bps (trading near post-2020 lows)\n`,
      `- **Credit Transmission:** Commercial lending rates remain elevated, rewarding cash-generative industrial and defense conglomerates.\n`,
      `- **Domestic Retail Liquidity:** Strong domestic equity positioning offsets real deposit rate competition.\n\n`,
      `## 2. Quantitative & Technical Findings\n\n`
    ];

    symbols.forEach((sym) => {
      reportTokens.push(
        `### ${sym} Indicator Structure\n`,
        `- **Trend Alignment:** The asset trades consistently above its 34-day and 89-day exponential moving averages, establishing a resilient medium-term regime.\n`,
        `- **Momentum Oscillators:** RSI currently prints in constructive territory without extreme overbought saturation; predicted 5-session forward RSI projects continuation towards the upper channel boundary.\n`,
        `- **Bollinger Bands:** Compression phase is resolving upward with volume confirmation observed across high-liquidity trading hours.\n\n`
      );
    });

    reportTokens.push(
      `## 3. Sentiment & News Risk Assessment\n\n`,
      `Our news analysis agent evaluated Turkish financial headlines, corporate disclosure filings (KAP), and cross-border sector dynamics over the trailing 14 calendar days:\n\n`,
      `- **Corporate Sentiment Index:** Net positive sentiment across industrial leaders, underpinned by robust export order books and backlog execution.\n`,
      `- **Regulatory & Tax Scrutiny:** No imminent transaction tax overhangs or structural margin curbs observed in latest official gazettes.\n\n`,
      `## 4. Multi-Agent Synthesis & Strategic Verdict\n\n`,
      `Cross-validation between technical trend signals and corporate disclosures confirms high model alignment. Portfolio managers are advised to maintain exposure with disciplined trailing stops pegged to the 34-day EMA, taking advantage of intraday liquidity pullbacks without chasing extended breakout levels.\n`
    );

    // Timed pipeline simulation (~14-15 seconds total)
    const timeline: Array<{ delay: number; action: () => void }> = [
      {
        delay: 400,
        action: () => {
          onEvent({ type: 'status', text: 'Verifying input files, dataset integrity and market sessions' });
        }
      },
      {
        delay: 1500,
        action: () => {
          onEvent({ type: 'status', text: 'Plan the analysis (supervisor)' });
          onEvent({ type: 'tool_call', agent: 'supervisor', tool: 'prediction_history' });
        }
      },
      {
        delay: 2600,
        action: () => {
          onEvent({ type: 'tool_result', agent: 'supervisor', tool: 'prediction_history' });
          onEvent({ type: 'status', text: 'Consulting technical_analyst' });
          onEvent({ type: 'tool_call', agent: 'technical_analyst', tool: 'technical_snapshot' });
        }
      },
      {
        delay: 4200,
        action: () => {
          onEvent({ type: 'tool_result', agent: 'technical_analyst', tool: 'technical_snapshot' });
          onEvent({ type: 'status', text: 'Consulting news_analyst' });
          onEvent({ type: 'tool_call', agent: 'news_analyst', tool: 'news_headlines' });
        }
      },
      {
        delay: 6000,
        action: () => {
          onEvent({ type: 'tool_result', agent: 'news_analyst', tool: 'news_headlines' });
          onEvent({ type: 'status', text: 'Consulting macro_analyst' });
          onEvent({ type: 'tool_call', agent: 'macro_analyst', tool: 'macro_snapshot' });
        }
      },
      {
        delay: 7800,
        action: () => {
          onEvent({ type: 'tool_result', agent: 'macro_analyst', tool: 'macro_snapshot' });
          if (hasDisclosures) {
            onEvent({ type: 'status', text: 'Consulting disclosure_analyst' });
            onEvent({ type: 'tool_call', agent: 'disclosure_analyst', tool: 'disclosure_document' });
          } else {
            // No disclosures, move directly
            onEvent({ type: 'status', text: 'Writing the report' });
          }
        }
      },
      {
        delay: 9200,
        action: () => {
          if (hasDisclosures) {
            onEvent({ type: 'tool_result', agent: 'disclosure_analyst', tool: 'disclosure_document' });
            onEvent({ type: 'status', text: 'Writing the report' });
          }
        }
      }
    ];

    // Schedule token streaming between 9.5s and 13.5s
    const tokenStartDelay = 9500;
    const tokenDuration = 4000;
    const tokenStep = Math.max(100, Math.floor(tokenDuration / reportTokens.length));

    reportTokens.forEach((token, idx) => {
      timeline.push({
        delay: tokenStartDelay + idx * tokenStep,
        action: () => {
          onEvent({ type: 'token', agent: 'supervisor', text: token });
        }
      });
    });

    // Final quality checks and final event at ~14.6s
    timeline.push({
      delay: 14200,
      action: () => {
        onEvent({ type: 'status', text: 'Performing quality checks & grounding verification' });
      }
    });

    timeline.push({
      delay: 14800,
      action: () => {
        const fullReport = reportTokens.join('');
        const technicalMap: Record<string, Record<string, string | number | boolean | null>> = {};
        const riskMap: Record<string, number> = {};

        symbols.forEach((sym) => {
          technicalMap[sym] = generateMockTechnical(sym);
          riskMap[sym] = sym === 'XU100' ? 2.8 : sym === 'THYAO' ? 3.2 : sym === 'ASELS' ? 2.6 : 4.1;
        });

        const finalEvent: StreamFinalEvent = {
          type: 'final',
          report: fullReport,
          technical: technicalMap,
          risk: riskMap,
          sources: [
            { name: 'Borsa İstanbul A.Ş.', attribution: 'BIST 100 Index & official market session feeds' },
            { name: 'İş Yatırım Ortaklığı', attribution: 'Historical equity adjusted price dataset & corporate actions' },
            { name: 'T.C. Merkez Bankası (CBRT)', attribution: 'EVDS monetary indicators & repo policy interest curves' },
            { name: 'Kamuoyu Aydınlatma Platformu (KAP)', attribution: 'Material event filings, quarterly disclosures & board resolutions' }
          ],
          timings: {
            total_seconds: 14.8
          },
          usage: {
            supervisor: { input_tokens: 3820, output_tokens: 1840, calls: 4 },
            technical_analyst: { input_tokens: 1420, output_tokens: 610, calls: symbols.length },
            news_analyst: { input_tokens: 1890, output_tokens: 720, calls: symbols.length },
            macro_analyst: { input_tokens: 1200, output_tokens: 430, calls: 1 },
            disclosure_analyst: { input_tokens: hasDisclosures ? 2100 : 0, output_tokens: hasDisclosures ? 510 : 0, calls: hasDisclosures ? 1 : 0 }
          },
          checks: {
            grounding_score: 0.984,
            checked_figures: 48 + symbols.length * 8,
            ungrounded_figures: [],
            unexpected_symbols: []
          },
          injection_flags: [],
          cached: false,
          analysis_id: `anl_${Date.now().toString(36)}`
        };

        incrementMockUsage(7420);
        onEvent(finalEvent);
        resolve();
      }
    });

    const timers: ReturnType<typeof setTimeout>[] = [];

    timeline.forEach((item) => {
      const t = setTimeout(() => {
        if (isAborted()) return;
        item.action();
      }, item.delay);
      timers.push(t);
    });

    if (signal) {
      signal.addEventListener('abort', () => {
        timers.forEach((t) => clearTimeout(t));
        reject(new DOMException('Analysis run aborted by user', 'AbortError'));
      });
    }
  });
}
