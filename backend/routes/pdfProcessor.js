// routes/pdfProcessor.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const fs = require('fs');
const pdfParse = require('pdf-parse');
const pinecone = require('@pinecone-database/pinecone');

// File upload config using multer
const upload = multer({ dest: 'uploads/' });

const pineconeClient = new pinecone.Pinecone({
  apiKey: 'pcsk_5ER3zb_6CoivEncEZMM2fUHSopCaL2tjByoehKgNxAybuYJg4ebRL877djhGqMm9u9X1r', 
});

let embedder;

// Load embedding model once
async function loadModel() {
  if (embedder) return;
  const { pipeline } = await import('@xenova/transformers');
  embedder = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2');
}

// Extract text from PDF
async function extractTextFromPDF(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = await pdfParse(dataBuffer);
  return pdfData.text;
}

// Store embeddings
async function storeEmbedding(id, text) {
  const embedderOutput = await embedder(text, { pooling: 'mean', normalize: true });
  const vector = Array.isArray(embedderOutput[0]) ? embedderOutput[0] : Array.from(embedderOutput[0]);

  if (vector.length !== 768) {
    throw new Error(`Vector dimension mismatch. Expected 768, got ${vector.length}`);
  }

  const index = pineconeClient.Index('crop-disesases');
  await index.upsert([{ id, values: vector, metadata: { text: text.slice(0, 200) } }]);
}

// Route: POST /api/process-pdf
router.post('/process-pdf', upload.single('pdf'), async (req, res) => {
  try {
    const file = req.file;
    const pdfId = req.body.id || `pdf-${Date.now()}`;

    if (!file) return res.status(400).json({ error: 'No PDF file uploaded' });

    await loadModel();
    const text = await extractTextFromPDF(file.path);
    await storeEmbedding(pdfId, text);

    // Clean up uploaded file
    fs.unlinkSync(file.path);

    res.status(200).json({ message: 'PDF processed and stored', id: pdfId });
  } catch (error) {
    console.error('Error processing PDF:', error);
    res.status(500).json({ error: 'Failed to process PDF' });
  }
});

module.exports = router;