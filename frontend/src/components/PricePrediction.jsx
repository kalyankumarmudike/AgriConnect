import React, { useState } from "react";
import axios from "axios";

const PricePrediction = () => {
  const districtMarketData = {
    Karimnagar: ['Dharmaram', 'Gangadhara', 'Husnabad', 'Huzzurabad', 'Jagtial', 'Manakodur', 'Manthani', 'Sircilla', 'Vemulawada'],
    Khammam: ['Bhadrachalam', 'Burgampadu', 'Charla', 'Dammapet', 'Kallur', 'Kothagudem', 'Wyra', 'Yellandu'],
    Mahbubnagar: ['Alampur', 'Devarakadra', 'Gadwal', 'Gadwal(Lezza)', 'Narayanpet', 'Wanaparthy Road(Prbbair)'],
    Nalgonda: ['Bhongir', 'Choutuppal', 'Devarakonda', 'Devarkonda(Dindi)', 'Devarkonda(Mallepalli)', 'Huzumnagar(Garidepally)', 'Huzurnagar', 'Huzurnagar(Matampally)', 'Kodad', 'Neredcherla', 'Nidamanoor', 'Suryapeta', 'Thungathurthy', 'Tirumalagiri'],
  };

  const varietyOptions = ['1001', 'I.R. 64', 'MTU-1010', 'Other', 'Samba Masuri', 'Paddy', 'Swarna Masuri (New)', 'Common', 'Hansa', 'Sona', 'B P T', 'HMT'];

  const [district, setDistrict] = useState("");
  const [market, setMarket] = useState("");
  const [variety, setVariety] = useState("");
  const [grade, setGrade] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    const data = { district, market, variety, grade };
    setLoading(true);
    setResult(null);

    try {
      const res = await axios.post("http://172.18.6.17:9500/predict", data);
      setResult(res.data?.prediction || {});
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setResult({ error: "Failed to get prediction" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto bg-white rounded shadow">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* District */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">District</label>
          <select
            value={district}
            onChange={(e) => {
              setDistrict(e.target.value);
              setMarket("");
            }}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="">Select District</option>
            {Object.keys(districtMarketData).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* Market */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Market</label>
          <select
            value={market}
            onChange={(e) => setMarket(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
            disabled={!district}
          >
            <option value="">Select Market</option>
            {district &&
              districtMarketData[district].map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
          </select>
        </div>

        {/* Variety */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Variety</label>
          <select
            value={variety}
            onChange={(e) => setVariety(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="">Select Variety</option>
            {varietyOptions.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        {/* Grade */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Grade</label>
          <select
            value={grade}
            onChange={(e) => setGrade(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="">Select Grade</option>
            <option value="FAQ">FAQ</option>
            <option value="Non-FAQ">Non-FAQ</option>
          </select>
        </div>
      </div>

      {/* Predict Button */}
      <button
        onClick={handleSubmit}
        className="px-4 py-2 bg-green-600 text-white rounded-md flex items-center justify-center gap-2"
        disabled={loading}
      >
        {loading && (
          <svg
            className="animate-spin h-4 w-4 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8H4z"
            ></path>
          </svg>
        )}
        {loading ? "Predicting results" : "Predict Results"}
      </button>

      {/* Result UI */}
      {result && !result.error && (
        <div className="mt-6 p-6 rounded-xl shadow-md bg-white border border-gray-200 max-w-md mx-auto">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">📈 Prediction Result</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">📅 Date:</span>
              <span className="font-medium text-gray-800">{result.date || "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">📍 Market:</span>
              <span className="font-medium text-gray-800">{result.market || "N/A"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">💰 Max Price:</span>
              <span className="font-semibold text-green-700">
                ₹{result.max_price !== undefined ? result.max_price.toFixed(2) : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">💸 Min Price:</span>
              <span className="font-semibold text-red-600">
                ₹{result.min_price !== undefined ? result.min_price.toFixed(2) : "N/A"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">⚖️ Modal Price:</span>
              <span className="font-semibold text-blue-700">
                ₹{result.modal_price !== undefined ? result.modal_price.toFixed(2) : "N/A"}
              </span>
            </div>
          </div>
        </div>
      )}

      {result?.error && (
        <p className="text-red-600 text-center mt-4">{result.error}</p>
      )}
    </div>
  );
};

export default PricePrediction;