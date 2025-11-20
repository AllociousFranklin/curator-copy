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
// 2️⃣ Supabase client (no loading embeddings anymore)
// ----------------------------
const supabaseUrl = 'https://lfonyzxytcdsvicymxor.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxmb255enh5dGNkc3ZpY3lteG9yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzU2ODQzNiwiZXhwIjoyMDc5MTQ0NDM2fQ.yVlE8KUFo-1_OcMQbfDgLHeZtQO8321ZX6lZN22Eb_I';
const supabase = createClient(supabaseUrl, supabaseKey);

// ----------------------------
// 3️⃣ Setup Gemini
// ----------------------------
const GEMINI_API_KEY = "AIzaSyC0LolRRAUetougA4djxd0oZJEMnwOMxIQ";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro" });

// ----------------------------
// 4️⃣ Query endpoint (now uses pgvector)
// ----------------------------
app.post("/query", async (req, res) => {
  try {
    const query = req.body.query;
    const queryEmbedding = req.body.embedding; // MUST come from client (MiniLM)

    if (!query) return res.status(400).send({ error: "Query is required" });
    if (!queryEmbedding) return res.status(400).send({ error: "Query embedding required" });

    console.log("🔍 User Query:", query);

    // Fetch most relevant chunks using Supabase pgvector
    const { data: topChunks, error } = await supabase.rpc("match_documents", {
      query_embedding: queryEmbedding,
      match_count: 10
    });

    if (error) {
      console.error("❌ Supabase RPC error:", error);
      return res.status(500).send({ error: "Supabase match_documents RPC failed" });
    }

    if (!topChunks || topChunks.length === 0) {
      return res.json({
        answer: { concise: "Not available in sources", detailed: "Not available in sources", sources: [] }
      });
    }

    const contextText = topChunks.map(c => c.text).join("\n\n");

    const prompt = `
You are a medical AI assistant.

Answer the following question using ONLY the retrieved context provided.

Respond strictly in this format:

Concise Answer: <max 2 lines>
Detailed Explanation: <5-8 lines with relevant medical details>
Sources: list top sources with short snippets

Context:
${contextText}

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
// 5️⃣ Start server
// ----------------------------
const PORT = 10000;
app.listen(PORT, () => console.log(`🚀 Server started on port ${PORT}`));
