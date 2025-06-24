// src/components/ImageUpload.js
import React, { useState, useRef } from 'react';
import axios from 'axios';
import { CameraIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline';

const ImageUpload = ({ onAnalysisComplete, onLoading, onError }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        
        // Create preview
        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(file);
      } else {
        onError('Please upload an image file.');
      }
    }
  };

  const openFileDialog = () => {
    fileInputRef.current.click();
  };

  const uploadImage = async () => {
    if (!selectedFile) {
      onError('Please select an image first.');
      return;
    }

    try {
      onLoading(true);
      onError(null);

      // Get user's location
      const position = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject);
      });

      const { latitude, longitude } = position.coords;

      // Create form data
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('latitude', latitude);
      formData.append('longitude', longitude);

      // Send to API
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Handle successful response
      onAnalysisComplete(response.data);
      onLoading(false);

      // If medium or high severity, show alert
      if (response.data.severity === 'medium' || response.data.severity === 'high') {
        alert(`Alert: Nearby farmers have been notified about this ${response.data.severity} severity ${response.data.disease} detection.`);
      }

    } catch (error) {
      console.error('Error uploading image:', error);
      onError(error.response?.data?.error || 'Failed to analyze image');
      onLoading(false);
    }
  };

  const captureImage = async () => {
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      
      // Create video element
      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();
      
      // Wait for video to be ready
      await new Promise(resolve => {
        video.onloadedmetadata = resolve;
      });
      
      // Create canvas to capture image
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      // Convert canvas to blob
      const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, 'image/jpeg');
      });
      
      // Create file from blob
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      
      // Set file and preview
      setSelectedFile(file);
      setPreview(URL.createObjectURL(blob));
      
      // Stop video stream
      stream.getTracks().forEach(track => track.stop());
      
    } catch (error) {
      console.error('Error capturing image:', error);
      onError('Failed to access camera');
    }
  };

  return (
    <div className="w-full">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
      />
      
      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center mb-4 transition-colors
          ${preview ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-green-500'}
        `}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        {preview ? (
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="max-h-64 mx-auto rounded-lg"
            />
                        <p className="mt-2 text-sm text-gray-500">Click to select a different image.</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center">
            <ArrowUpTrayIcon className="h-10 w-10 text-gray-400 mb-2" />
            <p className="text-gray-500">Drag & drop an image here, or click to select one.</p>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center gap-4">
        <button
          onClick={captureImage}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          <CameraIcon className="h-5 w-5" />
          Capture Image
        </button>
        <button
          onClick={uploadImage}
          className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition"
        >
          Analyze Image
        </button>
      </div>
    </div>
  );
};

export default ImageUpload;
