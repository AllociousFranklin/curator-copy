import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import fs from "fs";
import readline from "readline";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";

// ----------------------------
// 1️⃣ Express setup
// ----------------------------
const app = express();
app.use(cors());
app.use(bodyParser.json());

// ----------------------------
// 2️⃣ Load embeddings line-by-line (memory-efficient)
// ----------------------------
const embeddingsFile = "embeddings.jsonl";
const metadata = [];
const vectors = [];

console.log("📄 Loading embeddings.jsonl...");
try {
  const rl = readline.createInterface({
    input: fs.createReadStream(embeddingsFile),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    metadata.push({ id: obj.id, text: obj.text, source: obj.source });
    vectors.push(obj.embedding);
  }

  console.log(`✅ Loaded ${metadata.length} chunks.\n`);
} catch (err) {
  console.error("❌ Failed to load embeddings:", err);
}

// ----------------------------
// 3️⃣ Load MiniLM embedder
// ----------------------------
console.log("📦 Loading MiniLM embedder...");
const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
console.log("✅ MiniLM ready.\n");

// ----------------------------
// 4️⃣ Setup Gemini
// ----------------------------
const GEMINI_API_KEY = "AIzaSyC0LolRRAUetougA4djxd0oZJEMnwOMxIQ";
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
    // 5️⃣ Parse Gemini output (fixed for multiline)
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
const PORT = process.env.PORT || 10000;

app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));

