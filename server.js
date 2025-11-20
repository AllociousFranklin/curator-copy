import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from '@supabase/supabase-js';

// ----------------------------
// 1️⃣ Express setup
// ----------------------------
const app = express();
app.use(cors());
app.use(bodyParser.json());

// ----------------------------
// 2️⃣ Fetch embeddings from Supabase (hardcoded key for testing)
// ----------------------------
const supabaseUrl = 'https://lfonyzxytcdsvicymxor.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxmb255enh5dGNkc3ZpY3lteG9yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzU2ODQzNiwiZXhwIjoyMDc5MTQ0NDM2fQ.yVlE8KUFo-1_OcMQbfDgLHeZtQO8321ZX6lZN22Eb_I';
const supabase = createClient(supabaseUrl, supabaseKey);

const metadata = [];
const vectors = [];

console.log("📄 Fetching embeddings from Supabase...");
const { data: rows, error } = await supabase
  .from('embeddings')
  .select('*');

if (error) {
  console.error("❌ Supabase fetch error:", error);
} else {
  rows.forEach(row => {
    metadata.push({ id: row.id, text: row.text, source: row.source });
    vectors.push(row.embedding);
  });
  console.log(`✅ Loaded ${metadata.length} embeddings from Supabase.\n`);
}

// ----------------------------
// 3️⃣ Setup Gemini AI
// ----------------------------
const GEMINI_API_KEY = "AIzaSyC0LolRRAUetougA4djxd0oZJEMnwOMxIQ";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ----------------------------
// 4️⃣ Cosine similarity helper
// ----------------------------
function cosineSim(a, b) {
  let dot = 0, an = 0, bn = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    an += a[i] * a[i];
    bn += b[i] * b[i];
  }
  return dot / (Math.sqrt(an) * Math.sqrt(bn));
}

// ----------------------------
// 5️⃣ Top chunks helper
// ----------------------------
function getTopChunksPerDocument(queryEmbedding, topK = 3) {
  const perDoc = {};

  vectors.forEach((vec, idx) => {
    const score = cosineSim(queryEmbedding, vec);
    const src = metadata[idx].source;

    if (!perDoc[src]) perDoc[src] = [];
    perDoc[src].push({ score, text: metadata[idx].text });
  });

  const topChunks = [];
  Object.values(perDoc).forEach(chunks => {
    chunks.sort((a, b) => b.score - a.score);
    topChunks.push(...chunks.slice(0, topK));
  });

  topChunks.sort((a, b) => b.score - a.score);
  return topChunks.slice(0, 10);
}

// ----------------------------
// 6️⃣ Query endpoint
// ----------------------------
app.post("/query", async (req, res) => {
  try {
    const query = req.body.query;
    if (!query) return res.status(400).send({ error: "Query is required" });

    console.log("🔍 User Query:", query);

    // For testing, just use a dummy embedding or random vector
    // Replace this with real embedding API later
    const queryEmb = vectors[0] || Array(vectors[0].length).fill(0);

    const topChunks = getTopChunksPerDocument(queryEmb, 3);
    if (!topChunks.length) {
      return res.json({
        answer: { concise: "Not available in sources", detailed: "Not available in sources", sources: [] }
      });
    }

    const prompt = `
You are a medical AI assistant.

Answer the following question using ONLY the retrieved context provided.

Respond strictly in this format:

Concise Answer: <max 2 lines>
Detailed Explanation: <5-8 lines with relevant medical details>
Sources: list top sources with short snippets

Context:
${topChunks.map(c => c.text).join("\n\n")}

Question: ${query}
`;

    const aiResponse = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }]
    });

    const rawAnswer = aiResponse.response.text();
    console.log("Raw Gemini answer:", rawAnswer);

    let concise = "", detailed = "", sources = [];
    let currentSection = null;

    rawAnswer.split("\n").forEach(line => {
      const lower = line.toLowerCase();
      if (lower.startsWith("concise answer:")) {
        currentSection = "concise";
        concise = line.replace(/concise answer:/i, "").trim();
      } else if (lower.startsWith("detailed explanation:")) {
        currentSection = "detailed";
        detailed = line.replace(/detailed explanation:/i, "").trim();
      } else if (lower.startsWith("sources:")) {
        currentSection = "sources";
        sources = [];
      } else {
        if (currentSection === "concise") concise += " " + line.trim();
        else if (currentSection === "detailed") detailed += " " + line.trim();
        else if (currentSection === "sources" && line.trim()) sources.push(line.trim());
      }
    });

    res.json({ answer: { concise, detailed, sources } });
  } catch (err) {
    console.error("❌ RAG Error:", err);
    res.status(500).send({ error: "Internal Error: " + err.toString() });
  }
});

// ----------------------------
// 7️⃣ Start server
// ----------------------------
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));
  });
  console.log(`✅ Loaded ${metadata.length} embeddings from Supabase.\n`);
}

// ----------------------------
// 3️⃣ Load MiniLM embedder (Render-compatible embeddings pipeline)
// ----------------------------
console.log("📦 Loading MiniLM embedder...");
const embedder = await pipeline("embeddings", "Xenova/all-MiniLM-L6-v2");
console.log("✅ MiniLM ready.\n");

// ----------------------------
// 4️⃣ Setup Gemini
// ----------------------------
const GEMINI_API_KEY = "AIzaSyC0LolRRAUetougA4djxd0oZJEMnwOMxIQ"; // <-- Hardcoded Google Generative AI key
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ----------------------------
// 5️⃣ Cosine similarity helper
// ----------------------------
function cosineSim(a, b) {
  let dot = 0, an = 0, bn = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    an += a[i] * a[i];
    bn += b[i] * b[i];
  }
  return dot / (Math.sqrt(an) * Math.sqrt(bn));
}

// ----------------------------
// 6️⃣ Per-document top K helper
// ----------------------------
function getTopChunksPerDocument(queryEmbedding, topK = 3) {
  const perDoc = {};

  vectors.forEach((vec, idx) => {
    const score = cosineSim(queryEmbedding, vec);
    const src = metadata[idx].source;

    if (!perDoc[src]) perDoc[src] = [];
    perDoc[src].push({ score, text: metadata[idx].text });
  });

  const topChunks = [];
  Object.values(perDoc).forEach(chunks => {
    chunks.sort((a, b) => b.score - a.score);
    topChunks.push(...chunks.slice(0, topK));
  });

  topChunks.sort((a, b) => b.score - a.score);
  return topChunks.slice(0, 10); // overall top 10
}

// ----------------------------
// 7️⃣ Query endpoint
// ----------------------------
app.post("/query", async (req, res) => {
  try {
    const query = req.body.query;
    if (!query) return res.status(400).send({ error: "Query is required" });

    console.log("🔍 User Query:", query);

    // 1️⃣ Embed query
    const queryEmbRaw = await embedder(query, { pooling: "mean", normalize: true });
    const queryEmb = Array.from(queryEmbRaw.data);

    // 2️⃣ Retrieve top chunks
    const topChunks = getTopChunksPerDocument(queryEmb, 3);
    if (!topChunks.length) {
      return res.json({
        answer: { concise: "Not available in sources", detailed: "Not available in sources", sources: [] }
      });
    }

    console.log("Top chunks for context:", topChunks.map(c => c.text.slice(0, 100)));

    // 3️⃣ Build Gemini prompt
    const prompt = `
You are a medical AI assistant.

Answer the following question using ONLY the retrieved context provided.

Respond strictly in this format:

Concise Answer: <max 2 lines>
Detailed Explanation: <5-8 lines with relevant medical details>
Sources: list top sources with short snippets

Context:
${topChunks.map(c => c.text).join("\n\n")}

Question: ${query}
`;

    // 4️⃣ Call Gemini
    const aiResponse = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }]
    });
    const rawAnswer = aiResponse.response.text();
    console.log("Raw Gemini answer:", rawAnswer);

    // ----------------------------
    // 5️⃣ Parse Gemini output
    // ----------------------------
    let concise = "", detailed = "", sources = [];
    let currentSection = null;

    rawAnswer.split("\n").forEach(line => {
      const lower = line.toLowerCase();
      if (lower.startsWith("concise answer:")) {
        currentSection = "concise";
        concise = line.replace(/concise answer:/i, "").trim();
      } else if (lower.startsWith("detailed explanation:")) {
        currentSection = "detailed";
        detailed = line.replace(/detailed explanation:/i, "").trim();
      } else if (lower.startsWith("sources:")) {
        currentSection = "sources";
        sources = [];
      } else {
        if (currentSection === "concise") concise += " " + line.trim();
        else if (currentSection === "detailed") detailed += " " + line.trim();
        else if (currentSection === "sources" && line.trim()) sources.push(line.trim());
      }
    });

    res.json({ answer: { concise, detailed, sources } });

  } catch (err) {
    console.error("❌ RAG Error:", err);
    res.status(500).send({ error: "Internal Error: " + err.toString() });
  }
});

// ----------------------------
// 8️⃣ Start server
// ----------------------------
const PORT = 10000; // fixed port works fine on Render free tier
app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));
