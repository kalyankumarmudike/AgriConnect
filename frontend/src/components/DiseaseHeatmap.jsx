import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';

const DiseaseHeatmap = () => {
  const [heatmapData, setHeatmapData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    disease: '',
    severity: ''
  });
  const [bounds, setBounds] = useState([
    [17.37, 78.46], // Southwest coordinates
    [17.40, 78.50]  // Northeast coordinates
  ]);

  useEffect(() => {
    const fetchHeatmapData = async () => {
      try {
        setIsLoading(true);
        const params = new URLSearchParams();
        if (filters.disease) params.append('disease', filters.disease);
        if (filters.severity) params.append('severity', filters.severity);
        
        const response = await axios.get(`http://localhost:3000/heatmap?${params.toString()}`);
        setHeatmapData(response.data);
        setIsLoading(false);
      } catch (err) {
        setError(err.message);
        setIsLoading(false);
      }
    };

    fetchHeatmapData();
  }, [filters]);

  const getColorBySeverity = (severity) => {
    switch (severity) {
      case 'high': return '#ff0000'; // Red
      case 'medium': return '#ffa500'; // Orange
      case 'low': return '#00ff00'; // Green
      default: return '#0000ff'; // Blue
    }
  };

  const getRadiusByCount = (count) => {
    // More pronounced size difference
    return Math.min(10 + count * 5, 40);
  };

  if (isLoading) return <div className="loading">Loading heatmap data...</div>;
  if (error) return <div className="error">Error: {error}</div>;

  return (
    <div style={{ height: '100vh', width: '100%' }}>
      <div style={{ padding: '10px', backgroundColor: '#f0f0f0' }}>
        <h2>Crop Disease Heatmap</h2>
        <div style={{ marginBottom: '10px', display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          <div>
            <label>
              Filter by Disease:
              <select
                value={filters.disease}
                onChange={(e) => setFilters({...filters, disease: e.target.value})}
                style={{ marginLeft: '10px' }}
              >
                <option value="">All Diseases</option>
                <option value="Late Blight">Late Blight</option>
                <option value="Early Blight">Early Blight</option>
                <option value="Powdery Mildew">Powdery Mildew</option>
                <option value="Bacterial Spot">Bacterial Spot</option>
                <option value="Anthracnose">Anthracnose</option>
                <option value="Leaf Curl">Leaf Curl</option>
                <option value="Fusarium Wilt">Fusarium Wilt</option>
              </select>
            </label>
          </div>
          <div>
            <label>
              Filter by Severity:
              <select
                value={filters.severity}
                onChange={(e) => setFilters({...filters, severity: e.target.value})}
                style={{ marginLeft: '10px' }}
              >
                <option value="">All Severities</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <MapContainer
        center={[17.3850, 78.4867]} // Hyderabad coordinates
        zoom={14} // Increased zoom level
        style={{ height: '90%', width: '100%' }}
        bounds={bounds}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {heatmapData.map((dataPoint, index) => (
          <CircleMarker
            key={index}
            center={[dataPoint.location.coordinates[1], dataPoint.location.coordinates[0]]}
            radius={getRadiusByCount(dataPoint.count)}
            fillColor={getColorBySeverity(dataPoint.severity)}
            color="#000"
            weight={1}
            opacity={1}
            fillOpacity={0.7}
          >
            <Popup>
              <div style={{ minWidth: '200px' }}>
                <h4 style={{ margin: '5px 0', color: getColorBySeverity(dataPoint.severity) }}>
                  {dataPoint.disease}
                </h4>
                <p><strong>Severity:</strong> {dataPoint.severity}</p>
                <p><strong>Cases reported:</strong> {dataPoint.count}</p>
                <p><strong>Last reported:</strong> {new Date(dataPoint.timestamp).toLocaleString()}</p>
                <p><strong>Location:</strong> {dataPoint.location.coordinates[1].toFixed(4)}, {dataPoint.location.coordinates[0].toFixed(4)}</p>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
};

export default DiseaseHeatmap;