const mongoose = require('mongoose');

const HeatmapDataSchema = new mongoose.Schema({
  disease: String,
  severity: String,
  location: {
    type: {
      type: String,
      enum: ['Point'],
      required: true
    },
    coordinates: {
      type: [Number],
      required: true
    }
  },
  count: {
    type: Number,
    default: 1
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
});

HeatmapDataSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('heatmapData', HeatmapDataSchema);