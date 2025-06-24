require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');

const fs = require('fs');
const pdfParse = require('pdf-parse');
const pinecone = require('@pinecone-database/pinecone');

const Alert = require('./models/Alert');
const User = require('./models/User');
const HeatmapData = require('./models/HeatmapData');

const app = express();
const upload = multer();

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('MongoDB connection error:', err));

// Disease Prediction Endpoint
app.post('/storeDisease', async (req, res) => {
  try {
    const { disease, severity, latitude, longitude } = req.body;
    if (!latitude || !longitude) {
      return res.status(400).json({ error: 'Location coordinates are required for high/medium severity cases' });
    }

    // If severity is high or medium, create alert and notify users
    if (severity === 'high' || severity === 'medium') {
      
      // Create alert in database
      const alert = new Alert({
        disease,
        severity,
        location: {
          type: 'Point',
          coordinates: [parseFloat(longitude), parseFloat(latitude)]
        },
        affectedCrop: req.body.cropType || 'unknown'
      });

      await alert.save();

      // Notify nearby users
      await notifyNearbyUsers(alert);
      
      // Update heatmap data
      await updateHeatmapData(disease, severity, [parseFloat(longitude), parseFloat(latitude)]);
    }

    res.json(response.data);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ error: 'Failed to process prediction' });
  }
});

// Helper function to notify nearby users
async function notifyNearbyUsers(alert) {
  try {
    // Find users within 2km radius
    const users = await User.find({
      location: {
        $near: {
          $geometry: {
            type: "Point",
            coordinates: alert.location.coordinates
          },
          $maxDistance: alert.radius
        }
      },
      // Optional: filter by users who grow the affected crop
      // crops: alert.affectedCrop 
    });

    // Simulate sending notifications (implement actual notification service)
    const notificationPromises = users.map(async user => {
      // Here you would integrate with your actual notification service
      // (email, SMS, push notification, etc.)
      console.log(`Notifying user ${user.email} about ${alert.disease} outbreak`);
      
      // Add user to notifiedUsers array in alert
      alert.notifiedUsers.push(user._id);
    });

    await Promise.all(notificationPromises);
    await alert.save();

    console.log(`Notified ${users.length} users about disease alert`);
  } catch (error) {
    console.error('Error notifying users:', error);
  }
}

// Helper function to update heatmap data
async function updateHeatmapData(disease, severity, coordinates) {
  try {
    // Try to find existing heatmap data point nearby
    const existingData = await HeatmapData.findOneAndUpdate(
      {
        disease,
        severity,
        location: {
          $near: {
            $geometry: {
              type: "Point",
              coordinates
            },
            $maxDistance: 100 // 100 meters
          }
        }
      },
      { $inc: { count: 1 } },
      { new: true, upsert: true }
    );

    if (!existingData) {
      // Create new heatmap data point
      const newHeatmapData = new HeatmapData({
        disease,
        severity,
        location: {
          type: 'Point',
          coordinates
        }
      });
      await newHeatmapData.save();
    }
  } catch (error) {
    console.error('Error updating heatmap data:', error);
  }
}

// Get alerts endpoint
app.get('/alerts', async (req, res) => {
  try {
    const { latitude, longitude, radius = 2000 } = req.query;
    
    if (!latitude || !longitude) {
      return res.status(400).json({ error: 'Location coordinates are required' });
    }

    const alerts = await Alert.find({
      location: {
        $near: {
          $geometry: {
            type: "Point",
            coordinates: [parseFloat(longitude), parseFloat(latitude)]
          },
          $maxDistance: parseFloat(radius)
        }
      }
    }).sort({ timestamp: -1 });

    res.json(alerts);
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

// Test endpoint without filters
app.get('/heatmap-all', async (req, res) => {
  const allData = await HeatmapData.find({});
  res.json(allData);
});

// Get heatmap data endpoint
app.get('/heatmap', async (req, res) => {
  try {
    const { disease, severity } = req.query;
    const query = {};
    
    if (disease) query.disease = disease;
    if (severity) query.severity = severity;

    const heatmapData = await HeatmapData.find(query);
    console.log(heatmapData);
    
    res.json(heatmapData);
  } catch (error) {
    console.error('Error fetching heatmap data:', error);
    res.status(500).json({ error: 'Failed to fetch heatmap data' });
  }
});

app.get('/chatbot', async (req, res) => {
  try {
    
  } catch (error) {
    
  }
});






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
app.post('/process-pdf', upload.single('pdf'), async (req, res) => {
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

// Load embedding model
async function loadModel() {
  try {
    console.log('Loading model...');
    const { pipeline } = await import('@xenova/transformers');
    embedder = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2');
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

// Extract text from PDF
async function extractTextFromPDF(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const pdfData = await pdfParse(dataBuffer);
  return pdfData.text;
}

// Store embedding in Pinecone
async function storeEmbedding(id, text) {
  if (!embedder) throw new Error('Embedder model not loaded');

  const embedderOutput = await embedder(text, { pooling: 'mean', normalize: true });
  const vector = Array.isArray(embedderOutput[0]) ? embedderOutput[0] : Array.from(embedderOutput[0]);

  if (vector.length !== 768) {
    throw new Error(`Vector dimension mismatch. Expected 768, got ${vector.length}`);
  }

  const index = pineconeClient.Index('crop-disesases'); // replace with your index name
  await index.upsert([{ id: id, values: vector, metadata: { text: text.slice(0, 200) } }]);

  console.log(`Stored embedding for PDF with ID: ${id}`);
}

// API endpoint to handle PDF upload and processing
app.post('/upload', upload.single('pdf'), async (req, res) => {
  try {
    const pdfPath = req.file.path;
    const fileId = req.body.id || `pdf-${Date.now()}`;

    await loadModel();
    const text = await extractTextFromPDF(pdfPath);
    await storeEmbedding(fileId, text);

    fs.unlinkSync(pdfPath); // Delete temp file
    res.status(200).json({ message: 'PDF processed and stored successfully' });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error.message });
  }
});







// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});