"use client";

import { useState } from "react";
import { Sparkles, Trash2, ArrowRight } from "lucide-react";

interface PredictionResult {
  verdict: "positive" | "negative" | "neutral";
  confidence: number;
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
  script: string;
  script_label: string;
  cleaned_text: string;
}

const EXAMPLES = [
  "lah i3tik seha khouya zwin bzaf ❤️",
  "hadchi harban bzaf wallah 🔥",
  "تبارك الله عليك محتوى نقي بزاف",
  "hadchi 3yan bzaf flouss da3o 🤮",
  "bchhal taman dyal had l'produit?",
];

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyze = async (inputQuery?: string) => {
    const query = inputQuery !== undefined ? inputQuery : text;
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to analyze");
      }
      setResult(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Error analyzing text";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText("");
    setResult(null);
    setError(null);
  };

  const getVerdictStyle = (verdict: string) => {
    switch (verdict) {
      case "positive":
        return {
          bg: "bg-emerald-50 border-emerald-300 text-emerald-800",
          pill: "bg-emerald-600 text-white",
          dot: "🟢",
          title: "Positive",
          ar: "إيجابي",
        };
      case "negative":
        return {
          bg: "bg-rose-50 border-rose-300 text-rose-800",
          pill: "bg-rose-600 text-white",
          dot: "🔴",
          title: "Negative",
          ar: "سلبي",
        };
      default:
        return {
          bg: "bg-slate-100 border-slate-300 text-slate-800",
          pill: "bg-slate-600 text-white",
          dot: "⚪",
          title: "Neutral",
          ar: "محايد",
        };
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 flex flex-col items-center justify-center p-4 sm:p-6">
      <div className="w-full max-w-xl flex flex-col gap-6">
        {/* Simple Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🇲🇦</span>
            <h1 className="text-xl font-bold tracking-tight">Darija Sentiment</h1>
          </div>
          <span className="text-xs text-slate-600 font-medium bg-slate-100 px-2.5 py-1 rounded-full border border-slate-200">
            Arabic & Arabizi
          </span>
        </div>

        {/* Input Card */}
        <div className="bg-white rounded-2xl p-4 sm:p-5 shadow-sm border border-slate-200 flex flex-col gap-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                analyze();
              }
            }}
            placeholder="Type a comment in Moroccan Darija..."
            rows={3}
            className="w-full resize-none rounded-xl p-3 text-base text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 border border-slate-200"
          />

          {/* Quick Examples */}
          <div className="flex flex-wrap gap-1.5 items-center">
            <span className="text-xs text-slate-600 font-medium mr-1">Examples:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => {
                  setText(ex);
                  analyze(ex);
                }}
                className="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-lg transition-colors"
              >
                {ex}
              </button>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-1">
            <button
              type="button"
              onClick={clearAll}
              disabled={!text && !result}
              className="text-xs text-slate-600 hover:text-slate-700 disabled:opacity-30 flex items-center gap-1 px-2 py-1.5"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </button>

            <button
              type="button"
              onClick={() => analyze()}
              disabled={loading || !text.trim()}
              className="bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 text-white font-medium text-sm px-5 py-2 rounded-xl flex items-center gap-1.5 shadow-sm transition-all"
            >
              {loading ? (
                <span>Analyzing...</span>
              ) : (
                <>
                  <span>Analyze</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 text-rose-700 text-sm rounded-xl text-center">
            {error}
          </div>
        )}

        {/* Minimal Result Card */}
        {result && (
          <div
            className={`rounded-2xl p-5 border shadow-sm transition-all ${
              getVerdictStyle(result.verdict).bg
            }`}
          >
            {/* Header / Verdict */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2.5">
                <span className="text-2xl">{getVerdictStyle(result.verdict).dot}</span>
                <div>
                  <div className="text-xl font-bold flex items-center gap-2">
                    <span>{getVerdictStyle(result.verdict).title}</span>
                    <span className="text-base opacity-75">
                      ({getVerdictStyle(result.verdict).ar})
                    </span>
                  </div>
                  <span className="text-xs opacity-75 font-medium">
                    Script: {result.script_label}
                  </span>
                </div>
              </div>

              <div
                className={`text-sm font-bold px-3 py-1 rounded-full ${
                  getVerdictStyle(result.verdict).pill
                }`}
              >
                {Math.round(result.confidence * 100)}%
              </div>
            </div>

            {/* Simple Progress Bars */}
            <div className="space-y-2 pt-2 border-t border-black/5">
              {/* Positive */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1 text-emerald-900">
                  <span>Positive</span>
                  <span>{Math.round(result.scores.positive * 100)}%</span>
                </div>
                <div className="h-2 bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.positive * 100)}%` }}
                  />
                </div>
              </div>

              {/* Negative */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1 text-rose-900">
                  <span>Negative</span>
                  <span>{Math.round(result.scores.negative * 100)}%</span>
                </div>
                <div className="h-2 bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-rose-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.negative * 100)}%` }}
                  />
                </div>
              </div>

              {/* Neutral */}
              <div>
                <div className="flex justify-between text-xs font-semibold mb-1 text-slate-800">
                  <span>Neutral</span>
                  <span>{Math.round(result.scores.neutral * 100)}%</span>
                </div>
                <div className="h-2 bg-black/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-slate-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.neutral * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-xs text-slate-600">
          Trained on 51k+ Moroccan social comments
        </div>
      </div>
    </main>
  );
}
