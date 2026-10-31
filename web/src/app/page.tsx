"use client";

import { useState } from "react";
import { Trash2, ArrowRight } from "lucide-react";
import Logo from "@/components/Logo";

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
          bg: "bg-white border-zinc-200 text-zinc-900",
          pill: "bg-black text-white",
          dot: "🟢",
          title: "Positive",
          ar: "إيجابي",
          barColor: "bg-emerald-500",
        };
      case "negative":
        return {
          bg: "bg-white border-zinc-200 text-zinc-900",
          pill: "bg-rose-600 text-white",
          dot: "🔴",
          title: "Negative",
          ar: "سلبي",
          barColor: "bg-rose-500",
        };
      default:
        return {
          bg: "bg-white border-zinc-200 text-zinc-900",
          pill: "bg-zinc-700 text-white",
          dot: "⚪",
          title: "Neutral",
          ar: "محايد",
          barColor: "bg-zinc-400",
        };
    }
  };

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900 flex flex-col items-center justify-center p-4 sm:p-6 selection:bg-black selection:text-white">
      <div className="w-full max-w-xl flex flex-col gap-5">
        {/* Sleek Minimal Header with Logo */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Logo className="w-8 h-8" />
            <div>
              <h1 className="text-base font-semibold tracking-tight text-black leading-tight">
                Darija Sentiment
              </h1>
              <p className="text-[11px] text-zinc-600 font-medium">
                Moroccan Mixed-Script AI
              </p>
            </div>
          </div>
          <span className="text-xs text-zinc-600 font-medium bg-zinc-100 px-3 py-1 rounded-full border border-zinc-200">
            Arabic & Arabizi
          </span>
        </div>

        {/* Input Card */}
        <div className="bg-white rounded-2xl p-4 sm:p-5 shadow-sm border border-zinc-200 flex flex-col gap-3">
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
            className="w-full resize-none rounded-xl p-3 text-base text-zinc-800 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-black focus:border-black border border-zinc-200 transition-all"
          />

          {/* Quick Examples */}
          <div className="flex flex-wrap gap-1.5 items-center">
            <span className="text-xs text-zinc-600 font-medium mr-1">Examples:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => {
                  setText(ex);
                  analyze(ex);
                }}
                className="text-xs bg-zinc-100 hover:bg-zinc-200 active:bg-zinc-300 text-zinc-700 px-2.5 py-1 rounded-lg transition-colors"
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
              className="text-xs text-zinc-600 hover:text-zinc-900 disabled:opacity-30 flex items-center gap-1 px-2 py-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </button>

            <button
              type="button"
              onClick={() => analyze()}
              disabled={loading || !text.trim()}
              className="bg-black hover:bg-zinc-800 active:bg-zinc-900 disabled:opacity-40 text-white font-medium text-sm px-5 py-2 rounded-xl flex items-center gap-1.5 shadow-sm transition-all"
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
                  <div className="text-lg font-bold flex items-center gap-2 text-black">
                    <span>{getVerdictStyle(result.verdict).title}</span>
                    <span className="text-base text-zinc-600 font-normal">
                      ({getVerdictStyle(result.verdict).ar})
                    </span>
                  </div>
                  <span className="text-xs text-zinc-600 font-medium">
                    Script: {result.script_label}
                  </span>
                </div>
              </div>

              <div
                className={`text-xs font-bold px-3 py-1 rounded-full ${
                  getVerdictStyle(result.verdict).pill
                }`}
              >
                {Math.round(result.confidence * 100)}%
              </div>
            </div>

            {/* Simple Progress Bars */}
            <div className="space-y-2 pt-3 border-t border-zinc-100">
              {/* Positive */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1 text-zinc-700">
                  <span>Positive</span>
                  <span className="font-semibold text-zinc-900">
                    {Math.round(result.scores.positive * 100)}%
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.positive * 100)}%` }}
                  />
                </div>
              </div>

              {/* Negative */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1 text-zinc-700">
                  <span>Negative</span>
                  <span className="font-semibold text-zinc-900">
                    {Math.round(result.scores.negative * 100)}%
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-rose-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.negative * 100)}%` }}
                  />
                </div>
              </div>

              {/* Neutral */}
              <div>
                <div className="flex justify-between text-xs font-medium mb-1 text-zinc-700">
                  <span>Neutral</span>
                  <span className="font-semibold text-zinc-900">
                    {Math.round(result.scores.neutral * 100)}%
                  </span>
                </div>
                <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-zinc-400 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round(result.scores.neutral * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-xs text-zinc-600">
          Trained on 51k+ Moroccan social comments
        </div>
      </div>
    </main>
  );
}
