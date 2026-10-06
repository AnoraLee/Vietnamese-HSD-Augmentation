import { useEffect, useMemo, useState, useRef } from "react";
import { Boxes, ChartNoAxesColumnIncreasing, History, PanelLeft, Plus, ShieldAlert } from "lucide-react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const EXAMPLES = [
  { label: "Trung tính", text: "học kỳ cuối như đồ thị hình sóng thần" },
  { label: "Xúc phạm", text: "Hường lily mặt ngu vl" },
  { label: "Thù ghét", text: "mày khôn lắm thằng ngu ạ" },
];

const MODEL_OPTIONS = [
  { value: "phobert_baseline",  label: "PhoBERT Baseline",       shortLabel: "P-Base" },
  { value: "phobert_bt",        label: "PhoBERT Back-Translation", shortLabel: "P-BT" },
  { value: "phobert_eda",       label: "PhoBERT EDA",             shortLabel: "P-EDA" },
  { value: "phobert_llm",       label: "PhoBERT LLM-Gen",         shortLabel: "P-LLM" },
  { value: "phobert_combined",  label: "PhoBERT Combined",        shortLabel: "P-Comb" },
  { value: "visobert_baseline", label: "ViSoBERT Baseline",       shortLabel: "V-Base" },
  { value: "visobert_combined", label: "ViSoBERT Combined",       shortLabel: "V-Comb" },
];

const TOOL_OPTIONS = [
  {
    value: "classifier",
    label: "Phân loại độc hại",
    icon: ShieldAlert,
    title: "Bạn muốn kiểm tra gì?",
    description: "Phân tích nội dung tiếng Việt với các mô hình phát hiện độc hại và xem rõ điều gì tạo nên kết quả.",
  },
  {
    value: "history",
    label: "Lịch sử phân tích",
    icon: History,
    title: "Lịch sử phân tích",
    description: "Các phiên phân loại gần đây sẽ được hiển thị ở đây.",
  },
  {
    value: "models",
    label: "Các mô hình",
    icon: Boxes,
    title: "Các mô hình đang dùng",
    description: "Theo dõi các phiên bản PhoBERT và dữ liệu tăng cường được kết nối với workspace.",
  },
  {
    value: "evaluation",
    label: "Đánh giá kết quả",
    icon: ChartNoAxesColumnIncreasing,
    title: "Đánh giá kết quả",
    description: "So sánh độ chính xác, độ tin cậy và các lỗi thường gặp của từng mô hình.",
  },
];

const MODEL_CATALOG = [
  { value: "phobert_baseline",  name: "PhoBERT Baseline",  short: "P-BASE", path: "models/phobert_baseline_phobert",  accent: "blue",   note: "..." },
  { value: "phobert_bt",        name: "PhoBERT BT",         short: "P-BT",   path: "models/phobert_bt_phobert",        accent: "violet", note: "..." },
  { value: "phobert_eda",       name: "PhoBERT EDA",        short: "P-EDA",  path: "models/phobert_eda_phobert",       accent: "amber",  note: "..." },
  { value: "phobert_llm",       name: "PhoBERT LLM",        short: "P-LLM",  path: "models/phobert_llm_phobert",       accent: "pink",   note: "..." },
  { value: "phobert_combined",  name: "PhoBERT Combined",   short: "P-COMB", path: "models/phobert_combined_phobert",  accent: "green",  note: "..." },
  { value: "visobert_baseline", name: "ViSoBERT Baseline",  short: "V-BASE", path: "models/visobert_baseline_phobert", accent: "cyan",   note: "..." },
  { value: "visobert_combined", name: "ViSoBERT Combined",  short: "V-COMB", path: "models/visobert_combined_phobert", accent: "teal",   note: "..." },
];

const DEFAULT_MODEL = "phobert_combined";

// Error analysis -- vẫn hardcode vì backend chưa có endpoint cho phần này.
// TODO: chuyển sang /evaluation/errors?model=<exp> khi có file confusion_summary.
const COMBINED_ERRORS = [
  { from: "CLEAN", to: "OFFENSIVE", count: 946, tone: "offensive" },
  { from: "CLEAN", to: "HATE", count: 287, tone: "hate" },
  { from: "HATE", to: "OFFENSIVE", count: 156, tone: "offensive" },
  { from: "HATE", to: "CLEAN", count: 124, tone: "clean" },
  { from: "OFFENSIVE", to: "CLEAN", count: 115, tone: "clean" },
  { from: "OFFENSIVE", to: "HATE", count: 99, tone: "hate" },
];

const VERDICTS = {
  CLEAN: { name: "Không độc hại", gloss: "Không phát hiện dấu hiệu thù ghét hay xúc phạm.", className: "clean" },
  OFFENSIVE: { name: "Xúc phạm", gloss: "Có yếu tố thô tục hoặc mang tính công kích.", className: "offensive" },
  HATE: { name: "Thù ghét", gloss: "Chứa nội dung thù ghét — cần chú ý.", className: "hate" },
};

function apiErrorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) return payload.detail.message;
  return fallback;
}

const HISTORY_STORAGE_KEY = "hsd-analysis-history";

function readStoredHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) ?? "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

function tokenizeSegmented(text, importance) {
  if (!text) return [];
  const scores = (importance ?? []).map((entry) => entry?.score ?? 0);

  const positiveScores = scores.map((s) => Math.max(s, 0));
  const maxPositive = positiveScores.length
    ? Math.max(...positiveScores, 0.0001)
    : 0;

  return text.split(/\s+/).filter(Boolean).map((token, index) => {
    const raw = scores[index];
    const positive = positiveScores[index];
    const importanceValue =
      maxPositive && Number.isFinite(positive) && positive != null
        ? positive / maxPositive
        : 0;
    return {
      key: `${token}-${index}`,
      syllables: token.split("_"),
      compound: token.includes("_"),
      importance: Number.isFinite(importanceValue) ? importanceValue : 0,
      rawScore: Number.isFinite(raw) ? raw : 0,
    };
  });
}

function formatPercent(value) {
  const safe = Number.isFinite(value) ? value : 0;
  return new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 2 }).format(safe);
}

function formatHistoryDate(value) {
  return new Intl.DateTimeFormat("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

// CSS class-safe: "phobert_baseline" -> "phobert-baseline"
function cssKey(experiment) {
  return String(experiment ?? "").replace(/_/g, "-");
}

function HsdLogo() {
  return (
    <span className="hsd-logo" aria-hidden="true">
      <img src="/logo-mark.svg" alt="" draggable={false} />
    </span>
  );
}

function SidebarBrand() {
  return (
    <span className="sidebar-brand-text">
      <span className="brand-hate">Hate</span>
      <span className="brand-detect">Detect</span>
    </span>
  );
}

function ToolLogo({ type }) {
  if (type === "language") {
    return <svg viewBox="0 0 32 32" fill="none"><path d="M6 8h20v16H6z" rx="3" /><path d="M10 13h12M10 17h8M10 21h5" /></svg>;
  }
  if (type === "dataset") {
    return <svg viewBox="0 0 32 32" fill="none"><path d="M7 7h18v18H7z" rx="2" /><path d="M7 13h18M13 7v18M19 7v18" /></svg>;
  }
  if (type === "shap") {
    return <svg viewBox="0 0 32 32" fill="none"><path d="m7 23 7-8 5 4 7-10" /><circle cx="7" cy="23" r="2.5" /><circle cx="14" cy="15" r="2.5" /><circle cx="19" cy="19" r="2.5" /><circle cx="26" cy="9" r="2.5" /></svg>;
  }
  if (type === "model") {
    return <svg viewBox="0 0 32 32" fill="none"><circle cx="8" cy="16" r="3" /><circle cx="24" cy="9" r="3" /><circle cx="24" cy="23" r="3" /><path d="m11 15 10-5M11 17l10 5" /></svg>;
  }
  if (type === "api") {
    return <svg viewBox="0 0 32 32" fill="none"><path d="m12 7-7 9 7 9M20 7l7 9-7 9M18 5l-4 22" /></svg>;
  }
  return <svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="11" /><path d="M16 10v12M10 16h12" /></svg>;
}

function SidebarToggleIcon() {
  return <PanelLeft aria-hidden="true" />;
}

function ModelWorkspace({ activeModel, onSelectModel }) {
  return (
    <div className="workspace-view workspace-view--models">
      <div className="workspace-heading">
        <div>
          <span className="workspace-kicker">MODEL REGISTRY / 05 ACTIVE</span>
          <h1>Các mô hình đang dùng</h1>
          <p>Năm phiên bản PhoBERT được huấn luyện và sẵn sàng cho pipeline phân loại tiếng Việt.</p>
        </div>
        <div className="workspace-orbit" aria-hidden="true"><HsdLogo /></div>
      </div>
      <div className="model-grid">
        {MODEL_CATALOG.map((model) => (
          <button
            type="button"
            key={model.value}
            className={`model-card model-card--${model.accent}${activeModel === model.value ? " model-card--selected" : ""}`}
            onClick={() => onSelectModel(model.value)}
          >
            <div className="model-card-top">
              <span className="model-glyph">{model.short}</span>
              {activeModel === model.value && <span className="model-live">ĐANG DÙNG</span>}
            </div>
            <strong>{model.name}</strong>
            <span className="model-path">{model.path}</span>
            <small>{model.note}</small>
            <span className="model-card-arrow">↗</span>
          </button>
        ))}
      </div>
      <div className="model-footer-line"><span /> PhoBERT · Vietnamese HSD pipeline <span /></div>
    </div>
  );
}

function EvaluationWorkspace({ activeModel, metrics = [] }) {
  const largestError = COMBINED_ERRORS[0];

  if (!metrics.length) {
    return (
      <div className="workspace-view workspace-view--evaluation">
        <div className="workspace-heading">
          <div>
            <span className="workspace-kicker">EVALUATION LAB / TEST SET</span>
            <h1>Đánh giá kết quả</h1>
            <p>
              Chưa đọc được metrics. Chạy evaluation script và đặt file{" "}
              <code>results/metrics/metrics_&lt;experiment&gt;.json</code>.
            </p>
          </div>
        </div>
        <div className="tool-empty-note">
          Backend trả về 404 hoặc mảng rỗng từ <code>/evaluation/metrics</code>.
        </div>
      </div>
    );
  }

  const bestMacro = metrics.reduce(
    (best, item) => ((item.macroF1 ?? 0) > (best.macroF1 ?? 0) ? item : best),
    metrics[0],
  );
  const bestHate = metrics.reduce(
    (best, item) => ((item.hateF1 ?? 0) > (best.hateF1 ?? 0) ? item : best),
    metrics[0],
  );
  const selected =
    metrics.find((item) => item.experiment === activeModel) ?? metrics[0];

  return (
    <div className="workspace-view workspace-view--evaluation">
      <div className="workspace-heading">
        <div>
          <span className="workspace-kicker">EVALUATION LAB / TEST SET</span>
          <h1>Đánh giá kết quả</h1>
          <p>So sánh hiệu năng các thí nghiệm từ thư mục <code>results/</code> trên cùng tập kiểm thử.</p>
        </div>
        <div className="evaluation-stamp">LIVE<br /><b>RESULTS</b></div>
      </div>

      <div className={`metric-spotlight metric-spotlight--${cssKey(selected.experiment)}`}>
        <div>
          <span>MODEL ĐANG CHỌN</span>
          <strong>{selected.experiment.toUpperCase()}</strong>
          <small>kết quả theo model đang dùng</small>
        </div>
        <div>
          <span>ACCURACY</span>
          <strong>{formatPercent(selected.accuracy)}</strong>
          <small>test set</small>
        </div>
        <div>
          <span>MACRO-F1 TỐT NHẤT</span>
          <strong>{formatPercent(bestMacro.macroF1)}</strong>
          <small>{bestMacro.experiment.toUpperCase()}</small>
        </div>
        <div>
          <span>HATE-F1 TỐT NHẤT</span>
          <strong>{formatPercent(bestHate.hateF1)}</strong>
          <small>{bestHate.experiment.toUpperCase()}</small>
        </div>
      </div>

      <div className="evaluation-table-wrap">
        <div className="table-caption">
          <span>EXPERIMENT SUMMARY</span>
          <span>{metrics.length} RUNS · SORTED BY ACCURACY</span>
        </div>
        <div className="evaluation-table" role="table">
          <div className="evaluation-row evaluation-row--head" role="row">
            <span>EXPERIMENT</span>
            <span>ACCURACY</span>
            <span>MACRO-F1</span>
            <span>WEIGHTED-F1</span>
            <span>HATE-F1</span>
          </div>
          {[...metrics]
            .sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))
            .map((item) => (
              <div
                className={`evaluation-row evaluation-row--${cssKey(item.experiment)}${item.experiment === selected.experiment ? " evaluation-row--active" : ""}`}
                key={item.experiment}
                role="row"
              >
                <span>{item.experiment}</span>
                <span>{formatPercent(item.accuracy)}</span>
                <span>{formatPercent(item.macroF1)}</span>
                <span>{formatPercent(item.weightedF1)}</span>
                <span>{formatPercent(item.hateF1)}</span>
              </div>
            ))}
        </div>
      </div>

      <section className="error-analysis-panel">
        <div className="error-analysis-heading">
          <div>
            <span className="workspace-kicker">ERROR ANALYSIS / {selected.experiment.toUpperCase()}</span>
            <h2>Những lỗi mô hình hay gặp</h2>
          </div>
          <span className="error-analysis-source">confusion_summary.csv</span>
        </div>
        <div className="error-analysis-layout">
          <div className="error-callout">
            <span className="error-summary-icon">!</span>
            <div>
              <strong>Lỗi nổi bật nhất</strong>
              <p>
                <b>{largestError.from}</b> bị dự đoán thành <b>{largestError.to}</b>{" "}
                trong <em>{largestError.count.toLocaleString("vi-VN")}</em> trường hợp.
              </p>
            </div>
          </div>
          <div className="error-bars">
            {COMBINED_ERRORS.map((error) => (
              <div className="error-bar-row" key={`${error.from}-${error.to}`}>
                <span className="error-bar-label">{error.from} <i>→</i> {error.to}</span>
                <span className="error-bar-track">
                  <span
                    className={`error-bar-fill error-bar-fill--${error.tone}`}
                    style={{ width: `${(error.count / largestError.count) * 100}%` }}
                  />
                </span>
                <strong>{error.count.toLocaleString("vi-VN")}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

// ==========================================
// COMPONENT HIỂN THỊ TIN NHẮN CỦA AI
// ==========================================
function AiMessage({ message, metadata, onExplain }) {
  const [copied, setCopied] = useState(false);
  const { result, xai, isExplaining, xaiOpen, xaiError } = message;
  const verdict = VERDICTS[result.label] ?? VERDICTS.CLEAN;
  const labels = metadata?.labels ?? ["CLEAN", "OFFENSIVE", "HATE"];

  const tokenScores = xai?.token_scores ?? null;
  const tokens = useMemo(
    () => tokenizeSegmented(result.text_cleaned, tokenScores),
    [result.text_cleaned, tokenScores],
  );

  const features = xai?.token_scores ?? [];
  const maxAbs = features.length
    ? Math.max(...features.map((f) => Math.abs(f.score)), 0.0001)
    : 1;

  async function handleCopy() {
    if (!result?.text_cleaned) return;
    try {
      await navigator.clipboard.writeText(result.text_cleaned);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {}
  }

  return (
    <div className="chat-bubble ai-bubble">
      <div className="chat-avatar ai-avatar" aria-label="HSD Agent"><HsdLogo /></div>

      <div className="chat-content">
        <div className={`verdict verdict--${verdict.className}`}>
          <div className="verdict-head">
            <span className="verdict-eyebrow">Kết quả phân loại</span>
            <span className="verdict-meta">
              {formatPercent(result.confidence)} · {result.latency_ms.toFixed(0)} ms
            </span>
          </div>
          <p className="verdict-name">{verdict.name}</p>
          <p className="verdict-gloss">{verdict.gloss}</p>

          <dl className="ledger">
            {labels.map((label) => {
              const probability = result.probabilities[label] ?? 0;
              const entryClass = VERDICTS[label]?.className ?? "clean";
              return (
                <div className="ledger-row" key={label}>
                  <dt>{VERDICTS[label]?.name ?? label}</dt>
                  <dd>
                    <span className="ledger-track">
                      <span
                        className={`ledger-fill ledger-fill--${entryClass}`}
                        style={{ width: `${probability * 100}%` }}
                      />
                    </span>
                    <span className="ledger-value">{formatPercent(probability)}</span>
                  </dd>
                </div>
              );
            })}
          </dl>
        </div>

        <div className="reading-text">
          <p className="annotated">
            {tokens.map((token) => {
              const markVar =
                verdict.className === "hate" ? "--mark-hate"
                : verdict.className === "offensive" ? "--mark-offensive"
                : null;
              const style =
                markVar && token.importance > 0.05
                  ? {
                      backgroundColor: `rgba(var(${markVar}), ${(
                        0.12 + token.importance * 0.55
                      ).toFixed(2)})`,
                    }
                  : undefined;
              return (
                <span
                  key={token.key}
                  className={`token${token.compound ? " token--compound" : ""}`}
                  style={style}
                  title={
                    token.rawScore !== 0
                      ? `SHAP: ${token.rawScore >= 0 ? "+" : "−"}${Math.abs(token.rawScore).toFixed(3)}`
                      : undefined
                  }
                >
                  {token.syllables.join("_")}
                </span>
              );
            })}
          </p>
        </div>

        <div className={`xai-panel${xaiOpen ? " xai-panel--open" : ""}`}>
          <button
            type="button"
            className="xai-toggle"
            onClick={() => onExplain(message.id, result.text, result.model_used)}
            disabled={isExplaining}
            aria-expanded={xaiOpen}
          >
            <span className="xai-toggle-icon" aria-hidden="true">
              {isExplaining ? "⏳" : xaiOpen ? "▾" : "▸"}
            </span>
            <span className="xai-toggle-label">
              {isExplaining
                ? "Đang chạy SHAP…"
                : xai
                  ? "Giải thích bằng SHAP"
                  : "Xem từ ảnh hưởng đến kết quả"}
            </span>
            {!xai && !isExplaining && (
              <span className="xai-toggle-hint">SHAP</span>
            )}
          </button>

          {xaiError && (
            <p className="xai-error" role="alert">{xaiError}</p>
          )}

          {xai && xaiOpen && (
            <div className="xai-body">
              <div className="xai-heading">
                <div>
                  <p className="xai-kicker">Đóng góp của từng từ · SHAP</p>
                  <p className="xai-caption">
                    Những từ có ảnh hưởng đến nhãn{" "}
                    <strong style={{ color: `var(--${verdict.className})` }}>
                      {verdict.name}
                    </strong>
                    . Cột phải ủng hộ nhãn, cột trái làm giảm độ tin cậy.
                  </p>
                </div>
                <span className="xai-latency">{xai.latency_ms.toFixed(0)} ms</span>
              </div>

              <ul className="xai-chart">
                {features.map((f, i) => {
                  const pct = (Math.abs(f.score) / maxAbs) * 50;
                  const positive = f.score >= 0;
                  return (
                    <li key={`${f.token}-${i}`} className="xai-row">
                      <span className="xai-word" title={f.token}>{f.token}</span>
                      <span className="xai-track">
                        <span className="xai-axis" aria-hidden="true" />
                        <span
                          className={`xai-bar ${positive ? "xai-bar--pos" : "xai-bar--neg"}`}
                          style={{
                            width: `${pct}%`,
                            left: positive ? "50%" : undefined,
                            right: positive ? undefined : "50%",
                          }}
                        />
                      </span>
                      <span className="xai-value">
                        {f.score >= 0 ? "+" : "−"}
                        {Math.abs(f.score).toFixed(3)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        <div className="bubble-actions">
          <button type="button" className="ghost-btn" onClick={handleCopy}>
            {copied ? "✓ Đã copy" : "Copy văn bản"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// COMPONENT APP CHÍNH (GIAO DIỆN CHAT)
// ==========================================
export default function App() {
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [evaluationMetrics, setEvaluationMetrics] = useState([]);
  const [status, setStatus] = useState("checking");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [activeTool, setActiveTool] = useState("classifier");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [analysisHistory, setAnalysisHistory] = useState(readStoredHistory);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const lastMessageCount = useRef(0);

  const availableModelOptions = MODEL_OPTIONS.filter((option) =>
    metadata?.available_models?.includes(option.value) ?? true,
  );
  const selectedTool = TOOL_OPTIONS.find((tool) => tool.value === activeTool) ?? TOOL_OPTIONS[0];
  const isEmptyClassifier = activeTool === "classifier" && messages.length === 0;

  useEffect(() => {
    if (availableModelOptions.length && !availableModelOptions.some((option) => option.value === model)) {
      setModel(availableModelOptions[0].value);
    }
  }, [metadata, model, availableModelOptions.length]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "dark");
  }, []);

  useEffect(() => {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(analysisHistory));
  }, [analysisHistory]);

  useEffect(() => {
    if (messages.length > lastMessageCount.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
    lastMessageCount.current = messages.length;
  }, [messages, isSubmitting]);

  useEffect(() => {
    async function checkApi() {
      try {
        const response = await fetch(`${API_BASE_URL}/metadata`);
        const payload = await response.json();
        if (!response.ok) throw new Error("Lỗi kết nối");
        setMetadata(payload);
        setStatus("online");

        // Metrics là optional -- không block trạng thái online.
        try {
          const metricsRes = await fetch(`${API_BASE_URL}/evaluation/metrics`);
          if (metricsRes.ok) {
            const metrics = await metricsRes.json();
            setEvaluationMetrics(Array.isArray(metrics) ? metrics : []);
          }
        } catch {
          // Bỏ qua -- EvaluationWorkspace sẽ hiện empty state.
        }
      } catch {
        setStatus("offline");
      }
    }
    checkApi();
  }, []);

  async function handleSubmit(event) {
    if (event) event.preventDefault();
    if (!inputValue.trim() || isSubmitting) return;

    const userText = inputValue.trim();
    setInputValue("");

    if (textareaRef.current) textareaRef.current.style.height = "auto";

    const newMessageId = Date.now();
    setMessages((prev) => [...prev, { id: newMessageId, role: "user", text: userText }]);
    setIsSubmitting(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText, model }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(apiErrorMessage(payload, "Lỗi phân loại."));

      setAnalysisHistory((previous) => [
        {
          id: newMessageId,
          text: userText,
          label: payload.label,
          confidence: payload.confidence,
          model: payload.model_used ?? model,
          createdAt: new Date().toISOString(),
        },
        ...previous,
      ].slice(0, 30));

      setMessages((prev) => [
        ...prev,
        {
          id: newMessageId + 1,
          role: "ai",
          result: payload,
          xai: null,
          xaiOpen: false,
          isExplaining: false,
          xaiError: null,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: newMessageId + 1, role: "error", text: err.message },
      ]);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleExplain(messageId, textToExplain, modelUsed) {
    const currentMessage = messages.find((message) => message.id === messageId);
    if (!currentMessage) return;
    if (currentMessage.xai) {
      setMessages((prev) => prev.map((message) => (
        message.id === messageId
          ? { ...message, xaiOpen: !message.xaiOpen }
          : message
      )));
      return;
    }
    if (currentMessage.isExplaining) return;

    setMessages((prev) => prev.map((message) => (
      message.id === messageId
        ? { ...message, isExplaining: true, xaiOpen: true, xaiError: null }
        : message
    )));

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120_000);

    try {
      const response = await fetch(`${API_BASE_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToExplain, model: modelUsed, max_evals: 20 }),
        signal: controller.signal,
      });

      const raw = await response.text();
      let payload;
      try {
        payload = JSON.parse(raw);
      } catch {
        throw new Error(
          response.ok
            ? "Backend trả về dữ liệu không hợp lệ."
            : `Backend lỗi ${response.status}. Kiểm tra terminal uvicorn.`,
        );
      }

      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, `Backend lỗi ${response.status}.`));
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId ? { ...m, isExplaining: false, xai: payload, xaiError: null } : m,
        ),
      );
    } catch (err) {
      const msg = err.name === "AbortError"
        ? "Giải thích chạy quá lâu (>120s). Backend có thể đang treo."
        : err.message;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? { ...m, isExplaining: false, xaiOpen: false, xaiError: msg }
            : m,
        ),
      );
    } finally {
      clearTimeout(timeoutId);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  function handleInput(e) {
    setInputValue(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  }

  function handleNewChat() {
    setMessages([]);
    setInputValue("");
    setIsSubmitting(false);
    textareaRef.current?.focus();
  }

  return (
    <div className={`app-shell${isEmptyClassifier ? " app-shell--empty" : ""}${sidebarOpen ? "" : " app-shell--sidebar-collapsed"}`}>
      <aside className="app-sidebar" aria-label="Thanh điều hướng chính">
        <div className="sidebar-brand">
          <HsdLogo />
          <SidebarBrand />
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label={sidebarOpen ? "Thu gọn thanh bên" : "Mở rộng thanh bên"}
            aria-expanded={sidebarOpen}
          >
            <SidebarToggleIcon />
          </button>
        </div>

        <button type="button" className="new-chat-button" onClick={handleNewChat}>
          <Plus className="new-chat-icon" aria-hidden="true" />
          <span className="new-chat-label">Cuộc trò chuyện mới</span>
        </button>

        <div className="sidebar-section">
          <span className="sidebar-label">Công cụ</span>
          {TOOL_OPTIONS.map((tool) => (
            (() => {
              const ToolIcon = tool.icon;
              return (
                <button
                  type="button"
                  key={tool.value}
                  className={`sidebar-tool${activeTool === tool.value ? " active" : ""}`}
                  onClick={() => setActiveTool(tool.value)}
                  aria-current={activeTool === tool.value ? "page" : undefined}
                >
                  <span className="sidebar-tool-icon" aria-hidden="true"><ToolIcon /></span>
                  <span className="sidebar-tool-label">{tool.label}</span>
                </button>
              );
            })()
          ))}
        </div>

        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <span className="sidebar-note-dot" />
          <span className="sidebar-note-label">Chạy cục bộ · riêng tư</span>
        </div>
      </aside>

      <div className="chat-layout">
        <header className="chat-header">
          <div className="header-actions">
            <span className={`status-badge status--${status}`}>
              <span className="status-dot" aria-hidden="true" />
              {status === "online" ? "Sẵn sàng" : "Ngoại tuyến"}
            </span>
          </div>
        </header>

        <main className="chat-messages">
          {activeTool !== "classifier" || messages.length === 0 ? (
            activeTool === "models" ? <ModelWorkspace activeModel={model} onSelectModel={setModel} />
            : activeTool === "evaluation" ? <EvaluationWorkspace activeModel={model} metrics={evaluationMetrics} />
            : <div className={`empty-chat-greeting${activeTool !== "classifier" ? " tool-view" : ""}`}>
              <span className="tool-view-kicker">HSD / {activeTool.toUpperCase()}</span>
              <h1>{selectedTool.title}</h1>
              <p>{selectedTool.description}</p>
              {activeTool === "history" ? (
                <div className="history-list">
                  {analysisHistory.length ? analysisHistory.map((entry) => {
                    const verdict = VERDICTS[entry.label] ?? VERDICTS.CLEAN;
                    const modelLabel = MODEL_OPTIONS.find((option) => option.value === entry.model)?.label ?? entry.model;
                    return (
                      <article className="history-item" key={entry.id}>
                        <div className="history-item-main">
                          <span className={`history-status history-status--${verdict.className}`} />
                          <div>
                            <p className="history-text">{entry.text}</p>
                            <p className="history-meta">{formatHistoryDate(entry.createdAt)} · {modelLabel}</p>
                          </div>
                        </div>
                        <div className="history-result">
                          <strong className={`history-label history-label--${verdict.className}`}>{verdict.name}</strong>
                          <span>{formatPercent(entry.confidence)}</span>
                        </div>
                      </article>
                    );
                  }) : (
                    <div className="tool-empty-note">Chưa có phiên phân tích nào.</div>
                  )}
                </div>
              ) : activeTool === "classifier" ? (
                <>
                  <div className="tool-logo-marquee" aria-label="Các công cụ phân tích">
                    <div className="tool-logo-track">
                      {[0, 1].map((set) => (
                        <div className="tool-logo-set" key={set} aria-hidden={set === 1}>
                          <span className="tool-logo tool-logo--mail"><ToolLogo type="language" /></span>
                          <span className="tool-logo tool-logo--sheet"><ToolLogo type="dataset" /></span>
                          <span className="tool-logo tool-logo--shap"><ToolLogo type="shap" /></span>
                          <span className="tool-logo tool-logo--model"><ToolLogo type="model" /></span>
                          <span className="tool-logo tool-logo--api"><ToolLogo type="api" /></span>
                          <span className="tool-logo tool-logo--lab"><ToolLogo type="add" /></span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="greeting-examples">
                    {EXAMPLES.map(ex => (
                      <button key={ex.label} onClick={() => setInputValue(ex.text)} className="greeting-chip">
                        <span className="chip-label">Thử {ex.label.toLowerCase()}</span>
                        <span className="chip-text">"{ex.text}"</span>
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <div className="tool-empty-note">Chưa có dữ liệu trong workspace này.</div>
              )}
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg) => {
                if (msg.role === "user") {
                  return (
                    <div key={msg.id} className="chat-bubble user-bubble">
                      <div className="chat-content">{msg.text}</div>
                    </div>
                  );
                }
                if (msg.role === "error") {
                  return (
                    <div key={msg.id} className="chat-bubble ai-bubble error-bubble">
                      <div className="chat-avatar ai-avatar" aria-label="Lỗi">!</div>
                      <div className="chat-content error-text">{msg.text}</div>
                    </div>
                  );
                }
                return <AiMessage key={msg.id} message={msg} metadata={metadata} onExplain={handleExplain} />;
              })}

              {isSubmitting && (
                <div className="chat-bubble ai-bubble">
                  <div className="chat-avatar ai-avatar" aria-label="HSD Agent"><HsdLogo /></div>
                  <div className="chat-content thinking-indicator">
                    Đang xử lý
                    <span className="dots"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {activeTool === "classifier" ? <footer className="chat-footer">
          <div className="input-container">
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              value={inputValue}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder="Nhập văn bản cần kiểm tra..."
              rows={1}
            />
            <div className="input-bottom-row">
              {availableModelOptions.length === 1 ? (
                <div className="model-current" title={availableModelOptions[0].label}>
                  <span className="model-current-dot" aria-hidden="true" />
                  <span>{availableModelOptions[0].shortLabel}</span>
                  <span className="model-current-state">đang dùng</span>
                </div>
              ) : (
                <div
                  className="model-selector"
                  role="tablist"
                  aria-label="Chọn mô hình phân loại"
                  style={{ "--model-count": availableModelOptions.length }}
                >
                  <span
                    className="model-selector-indicator"
                    style={{ "--model-index": availableModelOptions.findIndex((option) => option.value === model) }}
                    aria-hidden="true"
                  />
                  {availableModelOptions.map((opt) => (
                    <button
                      type="button"
                      key={opt.value}
                      role="tab"
                      aria-selected={model === opt.value}
                      aria-label={opt.label}
                      title={opt.label}
                      className={`model-option${model === opt.value ? " model-option--active" : ""}`}
                      onClick={() => setModel(opt.value)}
                    >
                      {opt.shortLabel}
                    </button>
                  ))}
                </div>
              )}

              <button
                className="send-button"
                onClick={handleSubmit}
                disabled={!inputValue.trim() || isSubmitting || status !== "online"}
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M3.4 20.4L20.85 12.92C21.66 12.57 21.66 11.43 20.85 11.08L3.4 3.60002C2.74 3.31002 2.01 3.80002 2.01 4.51002L2 9.12002C2 9.62002 2.37 10.05 2.87 10.11L17 12L2.87 13.88C2.37 13.95 2 14.38 2 14.88L2.01 19.49C2.01 20.2 2.74 20.69 3.4 20.4Z" fill="currentColor"/>
                </svg>
              </button>
            </div>
          </div>
          <p className="footer-disclaimer">Kết quả chỉ mang tính tham khảo.</p>
        </footer> : (
          <footer className="chat-footer tool-footer">
            <span>Chọn "Phân loại độc hại" để bắt đầu một phiên kiểm tra mới.</span>
          </footer>
        )}
      </div>
    </div>
  );
}