import { useEffect, useMemo, useState, useRef } from "react";
import { ChartNoAxesColumnIncreasing, History, Moon, PanelLeft, ShieldAlert, Sun, X } from "lucide-react";

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
    value: "evaluation",
    label: "Đánh giá kết quả",
    icon: ChartNoAxesColumnIncreasing,
    title: "Đánh giá kết quả",
    description: "So sánh độ chính xác, độ tin cậy và các lỗi thường gặp của từng mô hình.",
  },
];

const DEFAULT_MODEL = "phobert_combined";

// Màu thanh lỗi lấy theo nhãn bị dự đoán nhầm thành.
const ERROR_TONES = { CLEAN: "clean", OFFENSIVE: "offensive", HATE: "hate" };

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
const THEME_STORAGE_KEY = "hsd-theme";

function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {}
  return "dark";
}

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

const METRIC_COLUMNS = [
  { id: "accuracy", label: "ACCURACY" },
  { id: "macroF1", label: "MACRO-F1" },
  { id: "weightedF1", label: "WEIGHTED-F1" },
  { id: "hateF1", label: "HATE-F1" },
];

function EvaluationWorkspace({ activeModel, metrics = [], errors = {} }) {
  // Hook phải nằm trên early return bên dưới.
  const [visibleColumns, setVisibleColumns] = useState(() => METRIC_COLUMNS.map((col) => col.id));
  // Model xem trên màn này, khởi tạo theo model đang dùng ở trình phân loại
  // nhưng đổi được độc lập -- xem kết quả model khác không phải đổi model đang chạy.
  const [pickedModel, setPickedModel] = useState(null);

  function toggleColumn(id) {
    setVisibleColumns((prev) => {
      if (!prev.includes(id)) {
        // Giữ nguyên thứ tự cột gốc khi bật lại.
        return METRIC_COLUMNS.filter((col) => prev.includes(col.id) || col.id === id).map((col) => col.id);
      }
      // Bỏ hết cột số thì bảng vô nghĩa -- luôn chừa lại một cột.
      return prev.length <= 1 ? prev : prev.filter((colId) => colId !== id);
    });
  }

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

  const selected =
    metrics.find((item) => item.experiment === (pickedModel ?? activeModel)) ?? metrics[0];
  const shownColumns = METRIC_COLUMNS.filter((col) => visibleColumns.includes(col.id));
  // Hạng của model đang xem theo từng chỉ số, để thay cho nhãn "tốt nhất" đã bỏ.
  function rankOf(field) {
    const order = [...metrics].sort((a, b) => (b[field] ?? 0) - (a[field] ?? 0));
    return order.findIndex((item) => item.experiment === selected.experiment) + 1;
  }

  // Confusion counts riêng của model đang chọn. Số loại lỗi khác nhau giữa các
  // model -- model không mắc loại nào thì CSV không có dòng đó.
  const selectedErrors = Array.isArray(errors[selected.experiment])
    ? errors[selected.experiment]
    : [];
  const largestError = selectedErrors[0] ?? null;

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

      <section className="error-analysis-panel">
        <div className="error-analysis-heading">
          <div>
            <span className="workspace-kicker">ERROR ANALYSIS / {selected.experiment.toUpperCase()}</span>
            <h2>Những lỗi mô hình hay gặp</h2>
          </div>
          <span className="error-analysis-source">confusion_summary.csv</span>
        </div>
        {largestError ? (
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
              {selectedErrors.map((error) => (
                <div className="error-bar-row" key={`${error.from}-${error.to}`}>
                  <span className="error-bar-label">{error.from} <i>→</i> {error.to}</span>
                  <span className="error-bar-track">
                    <span
                      className={`error-bar-fill error-bar-fill--${ERROR_TONES[error.to] ?? "offensive"}`}
                      style={{ width: `${(error.count / largestError.count) * 100}%` }}
                    />
                  </span>
                  <strong>{error.count.toLocaleString("vi-VN")}</strong>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="tool-empty-note">
            Chưa có dữ liệu lỗi cho <code>{selected.experiment}</code> trong{" "}
            <code>results/error_analysis/confusion_summary.csv</code>.
          </div>
        )}
      </section>

      <div className={`metric-spotlight metric-spotlight--${cssKey(selected.experiment)}`}>
        <div>
          <span>MODEL ĐANG CHỌN</span>
          <strong>{selected.experiment.toUpperCase()}</strong>
          <small>chọn một dòng trong bảng để đổi</small>
        </div>
        <div>
          <span>ACCURACY</span>
          <strong>{formatPercent(selected.accuracy)}</strong>
          <small>hạng {rankOf("accuracy")}/{metrics.length}</small>
        </div>
        <div>
          <span>MACRO-F1</span>
          <strong>{formatPercent(selected.macroF1)}</strong>
          <small>hạng {rankOf("macroF1")}/{metrics.length}</small>
        </div>
        <div>
          <span>HATE-F1</span>
          <strong>{formatPercent(selected.hateF1)}</strong>
          <small>hạng {rankOf("hateF1")}/{metrics.length}</small>
        </div>
      </div>

      <div className="evaluation-table-wrap">
        <div className="table-caption">
          <span>EXPERIMENT SUMMARY</span>
          <div className="column-filter" role="group" aria-label="Chọn cột hiển thị">
            <em>CỘT:</em>
            {METRIC_COLUMNS.map((col) => {
              const shown = visibleColumns.includes(col.id);
              const locked = shown && visibleColumns.length === 1;
              return (
                <button
                  type="button"
                  key={col.id}
                  className="column-chip"
                  aria-pressed={shown}
                  disabled={locked}
                  title={locked ? "Phải giữ ít nhất một cột" : shown ? "Ẩn cột" : "Hiện cột"}
                  onClick={() => toggleColumn(col.id)}
                >
                  {col.label}
                </button>
              );
            })}
          </div>
        </div>
        <div className="evaluation-table-scroll" style={{ "--metric-count": shownColumns.length }}>
        <div className="evaluation-table" role="table">
          <div className="evaluation-row evaluation-row--head" role="row">
            <span>EXPERIMENT</span>
            {shownColumns.map((col) => (
              <span key={col.id}>{col.label}</span>
            ))}
          </div>
          {[...metrics]
            .sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))
            .map((item) => (
              <div
                className={`evaluation-row evaluation-row--${cssKey(item.experiment)}${item.experiment === selected.experiment ? " evaluation-row--active" : ""}`}
                key={item.experiment}
                role="row"
                tabIndex={0}
                aria-selected={item.experiment === selected.experiment}
                title={`Xem kết quả của ${item.experiment}`}
                onClick={() => setPickedModel(item.experiment)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setPickedModel(item.experiment);
                  }
                }}
              >
                <span>{item.experiment}</span>
                {shownColumns.map((col) => (
                  <span key={col.id}>{formatPercent(item[col.id])}</span>
                ))}
              </div>
            ))}
        </div>
        </div>
      </div>
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
  const [evaluationErrors, setEvaluationErrors] = useState({});
  const [status, setStatus] = useState("checking");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [activeTool, setActiveTool] = useState("classifier");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [analysisHistory, setAnalysisHistory] = useState(readStoredHistory);
  const [theme, setTheme] = useState(readStoredTheme);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);

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
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch {}
  }, [theme]);

  function toggleTheme() {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }

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

        // Metrics và error analysis đều optional -- không block trạng thái online.
        try {
          const metricsRes = await fetch(`${API_BASE_URL}/evaluation/metrics`);
          if (metricsRes.ok) {
            const metrics = await metricsRes.json();
            setEvaluationMetrics(Array.isArray(metrics) ? metrics : []);
          }
        } catch {
          // Bỏ qua -- EvaluationWorkspace sẽ hiện empty state.
        }

        try {
          const errorsRes = await fetch(`${API_BASE_URL}/evaluation/errors`);
          if (errorsRes.ok) {
            const errors = await errorsRes.json();
            setEvaluationErrors(errors && typeof errors === "object" ? errors : {});
          }
        } catch {
          // Backend cũ chưa có /evaluation/errors -- khối Error Analysis sẽ báo thiếu dữ liệu.
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

      {/* Mobile drawer overlay */}
      <div className={`mobile-drawer-overlay${mobileDrawerOpen ? " open" : ""}`} onClick={() => setMobileDrawerOpen(false)} />
      <aside className={`mobile-drawer${mobileDrawerOpen ? " open" : ""}`} aria-label="Menu di động">
        <div className="mobile-drawer-header">
          <HsdLogo />
          <SidebarBrand />
          <button type="button" className="mobile-drawer-close" onClick={() => setMobileDrawerOpen(false)} aria-label="Đóng menu">
            <X size={20} />
          </button>
        </div>
        <div className="sidebar-section">
          <span className="sidebar-label">Công cụ</span>
          {TOOL_OPTIONS.map((tool) => {
            const ToolIcon = tool.icon;
            return (
              <button
                type="button"
                key={tool.value}
                className={`sidebar-tool${activeTool === tool.value ? " active" : ""}`}
                onClick={() => { setActiveTool(tool.value); setMobileDrawerOpen(false); }}
                aria-current={activeTool === tool.value ? "page" : undefined}
              >
                <span className="sidebar-tool-icon" aria-hidden="true"><ToolIcon /></span>
                <span className="sidebar-tool-label">{tool.label}</span>
              </button>
            );
          })}
        </div>
        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <span className="sidebar-note-dot" />
          <span className="sidebar-note-label">Chạy cục bộ · riêng tư</span>
        </div>
      </aside>

      <div className="chat-layout">
        <header className="chat-header">
          <div className="mobile-brand">
            <HsdLogo />
            <SidebarBrand />
          </div>
          <div className="header-actions">
            <span className={`status-badge status--${status}`}>
              <span className="status-dot" aria-hidden="true" />
              {status === "online" ? "Sẵn sàng" : "Ngoại tuyến"}
            </span>
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>

        <main className="chat-messages">
          {activeTool !== "classifier" || messages.length === 0 ? (
            activeTool === "evaluation" ? <EvaluationWorkspace activeModel={model} metrics={evaluationMetrics} errors={evaluationErrors} />
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
                <>
                  {/* Desktop: tab bar */}
                  <div
                    className="model-selector model-selector--desktop"
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
                  {/* Mobile: dropdown */}
                  <select
                    className="model-dropdown"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    aria-label="Chọn mô hình phân loại"
                  >
                    {availableModelOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </>
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
        </footer> : null}
      </div>

      {/* Mobile bottom navigation */}
      <nav className="mobile-bottom-nav" aria-label="Điều hướng nhanh">
        {TOOL_OPTIONS.map((tool) => {
          const ToolIcon = tool.icon;
          return (
            <button
              type="button"
              key={tool.value}
              className={`bottom-nav-item${activeTool === tool.value ? " active" : ""}`}
              onClick={() => setActiveTool(tool.value)}
              aria-current={activeTool === tool.value ? "page" : undefined}
            >
              <ToolIcon size={20} />
              <span>{tool.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}