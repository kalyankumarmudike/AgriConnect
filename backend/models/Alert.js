const mongoose = require('mongoose');

const AlertSchema = new mongoose.Schema({
  disease: {
    type: String,
    required: true
  },
  severity: {
    type: String,
    enum: ['low', 'medium', 'high'],
    required: true
  },
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
  radius: {
    type: Number,
    default: 2000 // 2km in meters
  },
  affectedCrop: String,
  timestamp: {
    type: Date,
    default: Date.now
  },
  notifiedUsers: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }]
});

AlertSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('Alert', AlertSchema);
