import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const EXAMPLES = [
  { label: "trung tính", text: "học kỳ cuối như đồ thị hình sóng thần" },
  { label: "xúc phạm", text: "Hường lily mặt ngu vl" },
  { label: "thù ghét", text: "mày khôn lắm thằng ngu ạ" },
];

const MODEL_OPTIONS = [
  { value: "baseline", label: "Baseline" },
  { value: "bt", label: "Back-Translation" },
  { value: "eda", label: "EDA" },
  { value: "llm", label: "LLM-Gen" },
  { value: "combined", label: "Combined" },
];

const VERDICTS = {
  CLEAN: {
    name: "Không độc hại",
    gloss: "Không phát hiện dấu hiệu thù ghét hay xúc phạm.",
    className: "clean",
  },
  OFFENSIVE: {
    name: "Xúc phạm",
    gloss: "Có yếu tố thô tục hoặc mang tính công kích.",
    className: "offensive",
  },
  HATE: {
    name: "Thù ghét",
    gloss: "Chứa nội dung thù ghét — cần chú ý.",
    className: "hate",
  },
};

function apiErrorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) return payload.detail.message;
  return fallback;
}

const THEME_STORAGE_KEY = "hsd-theme-preference";
const THEME_OPTIONS = [
  { value: "light", label: "Sáng" },
  { value: "dark", label: "Tối" },
  { value: "auto", label: "Tự động" },
];

/** 18:00–05:59 reads as "night" for the auto setting. */
function resolveAutoTheme(date = new Date()) {
  const hour = date.getHours();
  return hour >= 18 || hour < 6 ? "dark" : "light";
}

function readStoredThemePreference() {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) ?? "auto";
  } catch {
    return "auto";
  }
}

/** Split underthesea output into display tokens, flagging compound words
 * (joined by "_") so they can be rendered as a single visual unit —
 * this is the one piece of the pipeline that's genuinely specific to
 * Vietnamese, worth showing rather than hiding in a debug string.
 *
 * `importance` is the optional `token_importance` array from the API,
 * aligned 1:1 with the whitespace-split tokens of `text_cleaned`. Scores
 * are normalized against the sentence's own max so highlighting stays
 * legible regardless of the backend's raw attention scale. */
function tokenizeSegmented(text, importance) {
  if (!text) return [];
  const scores = (importance ?? []).map((entry) => entry?.score ?? 0);
  const maxScore = scores.length ? Math.max(...scores, 0.0001) : 0;

  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((token, index) => ({
      key: `${token}-${index}`,

      syllables: token.split("_"),
      compound: token.includes("_"),
      importance: maxScore ? (scores[index] ?? 0) / maxScore : 0,
    }));
}

function formatPercent(value) {
  return new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 2 }).format(value);
}

const HISTORY_LIMIT = 3;

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [status, setStatus] = useState("checking");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [history, setHistory] = useState([]);
  const [copied, setCopied] = useState(false);
  const [model, setModel] = useState("combined");
  const [themePreference, setThemePreference] = useState(readStoredThemePreference);
  const [autoTheme, setAutoTheme] = useState(() => resolveAutoTheme());

  const resolvedTheme = themePreference === "auto" ? autoTheme : themePreference;

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", resolvedTheme);
  }, [resolvedTheme]);

  useEffect(() => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, themePreference);
    } catch {
      // Riêng tư/incognito có thể chặn localStorage — bỏ qua, không chặn UI.
    }
  }, [themePreference]);

  // Chỉ cần với chế độ "auto": kiểm tra lại mỗi 5 phút để bắt đúng thời điểm
  // chuyển ngày/đêm (06:00, 18:00) nếu người dùng để trang mở xuyên qua mốc đó.
  useEffect(() => {
    if (themePreference !== "auto") return undefined;
    const interval = setInterval(() => setAutoTheme(resolveAutoTheme()), 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [themePreference]);

  const labels = useMemo(() => metadata?.labels ?? ["CLEAN", "OFFENSIVE", "HATE"], [metadata]);
  const tokens = useMemo(
    () => tokenizeSegmented(result?.text_cleaned, result?.token_importance),
    [result],
  );
  const verdict = result ? VERDICTS[result.label] ?? VERDICTS.CLEAN : null;

  useEffect(() => {
    async function checkApi() {
      try {
        const response = await fetch(`${API_BASE_URL}/metadata`);
        const payload = await response.json();
        if (!response.ok) throw new Error(apiErrorMessage(payload, "Backend chưa sẵn sàng."));
        setMetadata(payload);
        setStatus("online");
      } catch (requestError) {
        setStatus("offline");
        setError(requestError.message);
      }
    }
    checkApi();
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!text.trim()) {
      setError("Nhập một câu tiếng Việt trước khi phân loại.");
      return;
    }

    setError("");
    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(apiErrorMessage(payload, "Không thể phân loại văn bản."));
      setResult(payload);
      setStatus("online");
      setCopied(false);
      setHistory((previous) => [
        { id: `${Date.now()}`, text, label: payload.label, confidence: payload.confidence },
        ...previous,
      ].slice(0, HISTORY_LIMIT));
    } catch (requestError) {
      setStatus("offline");
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleCopyCleaned() {
    if (!result?.text_cleaned) return;
    try {
      await navigator.clipboard.writeText(result.text_cleaned);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Không thể copy — trình duyệt chặn quyền truy cập clipboard.");
    }
  }

  function handleTextareaKeyDown(event) {
    const isSubmitCombo = (event.metaKey || event.ctrlKey) && event.key === "Enter";
    if (isSubmitCombo) {
      event.preventDefault();
      handleSubmit(event);
    }
  }

  return (
    <main className="page">
      <header className="masthead">
        <div className="masthead-top">
          <div className="theme-picker" role="radiogroup" aria-label="Chọn giao diện sáng/tối">
            {THEME_OPTIONS.map((option) => (
              <button
                type="button"
                key={option.value}
                role="radio"
                aria-checked={themePreference === option.value}
                className={`theme-option${themePreference === option.value ? " theme-option--active" : ""}`}
                onClick={() => setThemePreference(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <h1>AI nhận diện văn bản độc hại tiếng Việt</h1>
        <p className="lede">
          Mô hình PhoBERT phân loại câu bạn nhập, thư viện underthesea đảm nhiệm việc phân từ tiếng Việt
          trước khi đưa vào mô hình.
        </p>
      </header>

      <form className="sheet" onSubmit={handleSubmit}>
        <div className="sheet-head">
          <label htmlFor="input-text">Nội dung cần phân loại</label>
          <span className={`status status--${status}`}>
            <span className="status-dot" aria-hidden="true" />
            {status === "online" ? "API sẵn sàng" : status === "checking" ? "Đang kết nối API" : "API chưa kết nối"}
          </span>
        </div>

        <div className="model-picker" role="radiogroup" aria-label="Chọn biến thể model">
          {MODEL_OPTIONS.map((option) => (
            <button
              type="button"
              key={option.value}
              role="radio"
              aria-checked={model === option.value}
              className={`model-option${model === option.value ? " model-option--active" : ""}`}
              onClick={() => setModel(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>

        <textarea
          id="input-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleTextareaKeyDown}
          maxLength={2000}
          rows={5}
          placeholder="Viết hoặc dán một câu — kể cả viết tắt, teencode…"
        />
        <p className="kbd-hint">
          <kbd>{navigator.platform.includes("Mac") ? "⌘" : "Ctrl"}</kbd> + <kbd>Enter</kbd> để phân loại nhanh
        </p>

        <div className="sheet-foot">
          <div className="examples">
            <span className="examples-label">Thử ngay</span>
            <div className="examples-buttons">
              {EXAMPLES.map((example) => (
                <button
                  type="button"
                  className="example-button"
                  key={example.label}
                  onClick={() => setText(example.text)}
                >
                  {example.label}
                </button>
              ))}
            </div>
          </div>
          <span className="char-count">{text.length}/2000</span>
        </div>

        <button className="submit" type="submit" disabled={isSubmitting || status !== "online"}>
          {isSubmitting ? "Đang phân tích…" : "Phân loại văn bản"}
        </button>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>

      {result && verdict && (
        <section className="reading" aria-live="polite">
          <div className="reading-text">
            <p className="annotated">
              {tokens.map((token) => {
                const markVar =
                  verdict.className === "hate"
                    ? "--mark-hate"
                    : verdict.className === "offensive"
                      ? "--mark-offensive"
                      : null;
                const style =
                  markVar && token.importance > 0.15
                    ? { backgroundColor: `rgba(var(${markVar}), ${(0.15 + token.importance * 0.55).toFixed(2)})` }
                    : undefined;
                return (
                  <span
                    className={`token${token.compound ? " token--compound" : ""}`}
                    key={token.key}
                    style={style}
                  >
                    {token.syllables.join("_")}
                  </span>
                );
              })}
            </p>
            <div className="reading-foot">
              <p className="reading-caption">
                Tách từ bằng Underthesea — nền cam/đỏ đánh dấu mức đóng góp vào nhãn dự đoán, gạch dưới
                đánh dấu từ ghép.
              </p>
              <button type="button" className="copy-button" onClick={handleCopyCleaned}>
                {copied ? "Đã copy" : "Copy văn bản đã tiền xử lý"}
              </button>
            </div>
          </div>

          <div className={`verdict verdict--${verdict.className}`}>
            <p className="verdict-eyebrow">Đã phân loại</p>
            <p className="verdict-name">{verdict.name}</p>
            <p className="verdict-gloss">{verdict.gloss}</p>
            <p className="verdict-confidence">
              Độ tin cậy {formatPercent(result.confidence)} · {result.latency_ms.toFixed(0)} ms
            </p>

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

            <p className="model-note">
              Mô hình {result.model_used}
              {metadata ? ` — ${metadata.preprocessing}` : ""}
            </p>
          </div>
        </section>
      )}

      {!result && (
        <section className="empty" aria-live="polite">
          <p>Chưa có gì để đọc. Nhập một câu ở trên rồi bấm "Phân loại văn bản".</p>
        </section>
      )}

      {history.length > 0 && (
        <section className="history">
          <p className="history-title">{HISTORY_LIMIT} lần gần nhất</p>
          <ul>
            {history.map((entry) => (
              <li key={entry.id} className={`history-item history-item--${VERDICTS[entry.label]?.className ?? "clean"}`}>
                <span className="history-text">{entry.text}</span>
                <span className="history-meta">
                  {VERDICTS[entry.label]?.name ?? entry.label} · {formatPercent(entry.confidence)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}