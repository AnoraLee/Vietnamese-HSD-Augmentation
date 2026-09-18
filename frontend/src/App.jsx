import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const EXAMPLES = [
  { label: "CLEAN", text: "học kỳ cuối như đồ thị hình sóng thần" },
  { label: "OFFENSIVE", text: "Hường lily mặt ngu vl" },
  { label: "HATE", text: "mày khôn lắm thằng ngu ạ" },
];

const LABEL_STYLES = {
  CLEAN: { icon: "✓", className: "clean" },
  OFFENSIVE: { icon: "!", className: "offensive" },
  HATE: { icon: "×", className: "hate" },
};

function apiErrorMessage(payload, fallback) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (payload?.detail?.message) return payload.detail.message;
  return fallback;
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [status, setStatus] = useState("checking");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const labels = useMemo(() => metadata?.labels ?? ["CLEAN", "OFFENSIVE", "HATE"], [metadata]);

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
      setError("Hãy nhập một câu tiếng Việt trước khi phân loại.");
      return;
    }

    setError("");
    setIsSubmitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(apiErrorMessage(payload, "Không thể phân loại văn bản."));
      setResult(payload);
      setStatus("online");
    } catch (requestError) {
      setStatus("offline");
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  const labelStyle = result ? LABEL_STYLES[result.label] ?? LABEL_STYLES.CLEAN : null;

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">PHOBERT · VNCORENLP</p>
          <h1>Phát hiện ngôn ngữ độc hại tiếng Việt</h1>
          <p className="subtitle">Phân tích văn bản trực tiếp với PhoBERT và pipeline phân từ tiếng Việt nhất quán với mô hình.</p>
        </div>
        <div className={`status ${status}`} role="status">
          <span className="status-dot" />
          {status === "online" ? "API sẵn sàng" : status === "checking" ? "Đang kiểm tra API" : "API chưa kết nối"}
        </div>
      </section>

      <section className="workspace" aria-label="Vietnamese hate speech classifier">
        <form className="input-card" onSubmit={handleSubmit}>
          <div className="card-heading">
            <div>
              <p className="section-kicker">VĂN BẢN ĐẦU VÀO</p>
              <h2>Nhập nội dung cần phân loại</h2>
            </div>
            <span className="character-count">{text.length}/2000</span>
          </div>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            maxLength={2000}
            placeholder="Nhập câu tiếng Việt: hệ thống hỗ trợ teencode và viết tắt..."
            aria-label="Văn bản tiếng Việt cần phân loại"
          />
          <div className="examples" aria-label="Ví dụ nhanh">
            <span>Thử nhanh:</span>
            {EXAMPLES.map((example) => (
              <button className="example-button" type="button" key={example.label} onClick={() => setText(example.text)}>
                {example.label}
              </button>
            ))}
          </div>
          <button className="submit-button" type="submit" disabled={isSubmitting || status !== "online"}>
            {isSubmitting ? "Đang phân tích…" : "Phân loại văn bản"}
          </button>
          {error && <p className="error-message" role="alert">{error}</p>}
        </form>

        <section className="result-card" aria-live="polite">
          {!result ? (
            <div className="empty-state">
              <span className="empty-icon">⌁</span>
              <h2>Kết quả sẽ xuất hiện ở đây</h2>
              <p>Nhập văn bản và chọn “Phân loại văn bản” để xem nhãn, xác suất và quá trình tiền xử lý.</p>
            </div>
          ) : (
            <>
              <div className="card-heading">
                <div>
                  <p className="section-kicker">KẾT QUẢ DỰ ĐOÁN</p>
                  <h2>Phân tích hoàn tất</h2>
                </div>
                <span className="latency">{result.latency_ms.toFixed(2)} ms</span>
              </div>

              <div className={`prediction ${labelStyle.className}`}>
                <span className="prediction-icon">{labelStyle.icon}</span>
                <div>
                  <p>Nhãn dự đoán</p>
                  <strong>{result.label}</strong>
                  <span>Độ tin cậy {new Intl.NumberFormat("vi-VN", { style: "percent", maximumFractionDigits: 2 }).format(result.confidence)}</span>
                </div>
              </div>

              <div className="probabilities">
                <h3>Phân bố xác suất</h3>
                {labels.map((label) => {
                  const probability = result.probabilities[label] ?? 0;
                  return (
                    <div className="probability-row" key={label}>
                      <span>{label}</span>
                      <div className="bar-track"><div className={`bar-fill ${LABEL_STYLES[label]?.className ?? ""}`} style={{ width: `${probability * 100}%` }} /></div>
                      <strong>{(probability * 100).toFixed(2)}%</strong>
                    </div>
                  );
                })}
              </div>

              <div className="preprocessing">
                <h3>Tiền xử lý VnCoreNLP</h3>
                <p>{result.text_cleaned}</p>
              </div>
              <p className="model-note">Model: {result.model_used}{metadata ? ` · ${metadata.preprocessing}` : ""}</p>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
