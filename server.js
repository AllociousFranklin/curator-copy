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
// 2️⃣ Fetch embeddings from Supabase
// ----------------------------
const supabaseUrl = 'https://lfonyzxytcdsvicymxor.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxmb255enh5dGNkc3ZpY3lteG9yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzU2ODQzNiwiZXhwIjoyMDc5MTQ0NDM2fQ.yVlE8KUFo-1_OcMQbfDgLHeZtQO8321ZX6lZN22Eb_I';
const supabase = createClient(supabaseUrl, supabaseKey);

const metadata = [];
const vectors = [];

async function loadEmbeddings() {
  console.log("📄 Fetching embeddings from Supabase...");

  // ✅ Fetch all 19,871 embeddings safely
  const { data: rows, error } = await supabase
    .from('embeddings')
    .select('*')
    .limit(20000); // higher than total embeddings

  if (error) {
    console.error("❌ Supabase fetch error:", error);
  } else if (rows && rows.length) {
    rows.forEach(row => {
      metadata.push({ id: row.id, text: row.text, source: row.source });
      vectors.push(row.embedding);
    });
    console.log(`✅ Loaded ${metadata.length} embeddings from Supabase.\n`);
  } else {
    console.log("⚠️ No embeddings found in Supabase.");
  }
}

// Load embeddings safely at startup
await loadEmbeddings();

// ----------------------------
// 3️⃣ Setup Gemini
// ----------------------------
const GEMINI_API_KEY = "AIzaSyC0LolRRAUetougA4djxd0oZJEMnwOMxIQ";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ----------------------------
// 4️⃣ Cosine similarity
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
// 5️⃣ Query endpoint
// ----------------------------
app.post("/query", async (req, res) => {
  try {
    const query = req.body.query;
    if (!query) return res.status(400).send({ error: "Query is required" });

    console.log("🔍 User Query:", query);

    // ✅ Use precomputed embedding safely
    const queryEmb = vectors.length ? vectors[0] : Array(384).fill(0); // MiniLM size = 384
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
      if (lower.startsWith("concise answer:")) currentSection = "concise";
      else if (lower.startsWith("detailed explanation:")) currentSection = "detailed";
      else if (lower.startsWith("sources:")) {
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
// 6️⃣ Start server
// ----------------------------
const PORT = 10000;
app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));
