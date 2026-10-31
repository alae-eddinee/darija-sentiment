import weights from "./model_weights.json";

interface SentimentResult {
  verdict: "positive" | "negative" | "neutral";
  confidence: number;
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
  script: "arabic" | "arabizi" | "mixed";
  script_label: string;
  cleaned_text: string;
}

export function cleanText(text: string): string {
  if (!text) return "";
  let s = text.replace(/https?:\/\/\S+/g, "").replace(/www\.\S+/g, "");
  s = s.replace(/@\w+/g, "");
  // Collapse repeated characters to max 2
  s = s.replace(/(.)\1{2,}/g, "$1$1");
  return s.replace(/\s+/g, " ").trim();
}

export function detectScript(text: string): "arabic" | "arabizi" | "mixed" {
  const arabicRegex = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/g;
  const latinRegex = /[a-zA-Z]/g;
  const arabiziDigits = /[23579]/g;

  const totalChars = text.replace(/\s+/g, "").length;
  if (totalChars === 0) return "arabic";

  const arMatches = (text.match(arabicRegex) || []).length;
  const latMatches = (text.match(latinRegex) || []).length;
  const digitMatches = (text.match(arabiziDigits) || []).length;

  const arRatio = arMatches / totalChars;
  const latRatio = latMatches / totalChars;

  if (arRatio > 0.4 && latRatio < 0.2) return "arabic";
  if (latRatio > 0.3 && arRatio > 0.2) return "mixed";
  if (digitMatches > 0 && latRatio > 0.2) return "arabizi";
  if (latRatio > 0.4) return "arabizi";

  return arRatio >= latRatio ? "arabic" : "arabizi";
}

export function predictSentiment(rawText: string): SentimentResult {
  const cleaned = cleanText(rawText);
  const script = detectScript(rawText);

  const tokens = weights.tokens as unknown as Record<string, [number, number[]]>;
  const intercept = weights.intercept as unknown as number[];

  // 1. Extract term counts for char n-grams (2-5) and word n-grams (1-2)
  const termCounts: Record<string, number> = {};

  // Words
  const words = cleaned.toLowerCase().split(/\s+/).filter(Boolean);

  // Word 1-grams & 2-grams
  for (let i = 0; i < words.length; i++) {
    const w1 = "w:" + words[i];
    termCounts[w1] = (termCounts[w1] || 0) + 1;

    if (i < words.length - 1) {
      const w2 = "w:" + words[i] + " " + words[i + 1];
      termCounts[w2] = (termCounts[w2] || 0) + 1;
    }
  }

  // Char_wb 2-5 grams
  for (const w of words) {
    const padded = " " + w + " ";
    for (let n = 2; n <= 5; n++) {
      for (let i = 0; i <= padded.length - n; i++) {
        const sub = "c:" + padded.substring(i, i + n);
        termCounts[sub] = (termCounts[sub] || 0) + 1;
      }
    }
  }

  // 2. Compute TF-IDF features with sublinear TF & L2 normalization
  // sublinear_tf: tf = 1 + log(count)
  let sumSq = 0;
  const activeFeatures: Array<{ coefs: number[]; val: number }> = [];

  for (const [key, count] of Object.entries(termCounts)) {
    const entry = tokens[key];
    if (!entry) continue;

    const [idf, coefs] = entry;
    const tf = 1 + Math.log(count);
    const tfidf = tf * idf;

    sumSq += tfidf * tfidf;
    activeFeatures.push({ coefs, val: tfidf });
  }

  // L2 normalize
  const norm = Math.sqrt(sumSq) || 1.0;

  // 3. Dot product with classifier weights: dot(X, coef.T) + intercept
  // classes: ['negative', 'neutral', 'positive']
  const scores = [...intercept];

  for (const feat of activeFeatures) {
    const normVal = feat.val / norm;
    for (let c = 0; c < 3; c++) {
      scores[c] += feat.coefs[c] * normVal;
    }
  }

  // 4. Softmax
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - maxScore));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);

  const classes: Array<"negative" | "neutral" | "positive"> = [
    "negative",
    "neutral",
    "positive",
  ];
  let maxIdx = 0;
  for (let i = 1; i < 3; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }

  const scriptLabels = {
    arabic: "Arabic Script",
    arabizi: "Arabizi (3, 7, 9)",
    mixed: "French Code-Switching",
  };

  return {
    verdict: classes[maxIdx],
    confidence: Math.round(probs[maxIdx] * 1000) / 1000,
    scores: {
      negative: Math.round(probs[0] * 1000) / 1000,
      neutral: Math.round(probs[1] * 1000) / 1000,
      positive: Math.round(probs[2] * 1000) / 1000,
    },
    script,
    script_label: scriptLabels[script],
    cleaned_text: cleaned,
  };
}
