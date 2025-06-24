const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  name: String,
  email: {
    type: String,
    unique: true
  },
  phone: String,
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
  crops: [String], // Types of crops the user grows
  notificationPreferences: {
    email: Boolean,
    sms: Boolean,
    push: Boolean
  },
  deviceToken: String // For push notifications
});

UserSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('User', UserSchema);
