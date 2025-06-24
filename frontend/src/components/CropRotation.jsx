import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MapPin,
  Loader,
  Droplets,
  Zap,
  Leaf,
  AlertTriangle,
  Calendar,
  Wind,
  WindIcon,
  Thermometer,
  SproutIcon,
  CloudSunIcon,
  LightbulbIcon,
  LeafIcon,
  LayersIcon,
  CalendarClock,
  FlaskConical,
  ChevronDown,
  ChevronUp,
  Info,
  Droplet,
  Sun,
  Clock,
} from "lucide-react";
import axios from "axios";
import MultiStepLoader from "../reusableComponents/MultiStepLoader.jsx";
import { soilTypes, previousCrops, commonCrops } from "../data.js";
import AgricultureRecommendations from "../reusableComponents/AgricultureRecommendations.jsx";

export default function CropPlanningSystem() {
  const [formData, setFormData] = useState({
    latitude: "",
    longitude: "",
    soilType: "loamy",
    nitrogen: 50,
    phosphorus: 25,
    potassium: 75,
    ph: 6.5,
    organicMatter: 2.0,
    previousCrop: "",
    specificCrop: "",
    useIoTData: false,
  });

  const [selectedOption, setSelectedOption] = useState("");
  const [loading, setLoading] = useState(false);
  const [fetchingLocation, setFetchingLocation] = useState(false);
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(1);
  const [selectedCrop, setSelectedCrop] = useState(null);

  const toggleCropSelection = (crop) => {
    setSelectedCrop(selectedCrop === crop ? null : crop);
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const getLocation = () => {
    setFetchingLocation(true);

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setFormData((prev) => ({
            ...prev,
            latitude: position.coords.latitude.toFixed(4),
            longitude: position.coords.longitude.toFixed(4),
          }));
          setFetchingLocation(false);
        },
        (error) => {
          alert("Unable to retrieve your location. Please enter manually.");
          setFetchingLocation(false);
        }
      );
    } else {
      alert(
        "Geolocation is not supported by this browser. Please enter location manually."
      );
      setFetchingLocation(false);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);

    // Prepare API request body
    const requestBody = {
      latitude: parseFloat(formData.latitude),
      longitude: parseFloat(formData.longitude),
      soil_type:
        formData.soilType.charAt(0).toUpperCase() +
        formData.soilType.slice(1) +
        " Soil",
      nitrogen: parseFloat(formData.nitrogen),
      phosphorus: parseFloat(formData.phosphorus),
      potassium: parseFloat(formData.potassium),
      ph: parseFloat(formData.ph),
      organic_matter: parseFloat(formData.organicMatter),
      previous_crop:
        formData.previousCrop.charAt(0).toUpperCase() +
        formData.previousCrop.slice(1),
    };

    // If specific crop selected
    if (selectedOption === "specific" && formData.specificCrop) {
      requestBody.specific_crop = formData.specificCrop;
    }

    try {
      // Simulate API call with a delay
      setTimeout(() => {
        // Mock result data
        // const mockResult = {
        //   recommendations: "**Executive Summary:** Based on your soil analysis and location data, we recommend cultivating the following crops for optimal yield and sustainability.\n\n**Rice:** Suitable due to your soil composition and previous crop history.\n\n**Soybean:** Good option for nitrogen fixation and market value.\n\n**Maize:** Compatible with your climate zone and potassium levels.",
        //   weather_summary: {
        //     min_temp: 22,
        //     max_temp: 35,
        //     total_precip: 450,
        //     avg_wind: 8,
        //     period: "May - September"
        //   },
        //   insights: [
        //     "✅ Soil fertility is optimal for grain crops",
        //     "⚠️ Consider additional phosphorus supplements",
        //     "✅ Climate conditions favorable for selected crops",
        //     "✅ Crop rotation strategy is appropriate"
        //   ],
        //   npk_values: {
        //     N: formData.nitrogen,
        //     P: formData.phosphorus,
        //     K: formData.potassium,
        //     pH: formData.ph,
        //     organic_matter: formData.organicMatter
        //   }
        // };

        axios
          .post("http://127.0.0.1:5000/cropRotation", requestBody)
          .then((response) => {
            // Handle the successful response here
            console.log("Crop rotation suggestions:", response.data);
            setResult(response.data);
            setLoading(false);
            setStep(3);
            // You might want to update your component's state with the response data
          })
          .catch((error) => {
            // Handle any errors that occurred during the request
            console.error("Error fetching crop rotation:", error);
            alert("Failed to get recommendations. Please try again.");
            setLoading(false);
            // You might want to display an error message to the user
          });
      }, 3000); // 3 second delay to show the loader
    } catch (error) {
      console.error("Error fetching data:", error);
      alert("Failed to get recommendations. Please try again.");
      setLoading(false);
    }
  };

  const nextStep = () => {
    if (step === 1) {
      if (
        !formData.latitude ||
        !formData.longitude ||
        !formData.previousCrop ||
        !formData.soilType
      ) {
        alert("Please fill all required fields before proceeding.");
        return;
      }
    }

    if (step === 2) {
      if (selectedOption === "specific" && !formData.specificCrop) {
        alert("Please select a specific crop.");
        return;
      }
      handleSubmit();
    } else {
      setStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    setStep((prev) => prev - 1);
  };

  // Render form steps
  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="col-span-2">
                <h3 className="text-lg font-semibold text-green-800 mb-4">
                  Location & Field Details
                </h3>
              </div>

              <div>
                <label className="block text-sm font-medium text-green-800 mb-1">
                  Latitude & Longitude
                </label>
                <div className="flex items-center">
                  <div className="flex-1 mr-2">
                    <input
                      type="text"
                      name="latitude"
                      value={formData.latitude}
                      onChange={handleChange}
                      className="w-full p-2 border border-green-500 rounded-md bg-white text-green-800"
                      placeholder="Latitude"
                    />
                  </div>
                  <div className="flex-1 mr-2">
                    <input
                      type="text"
                      name="longitude"
                      value={formData.longitude}
                      onChange={handleChange}
                      className="w-full p-2 border border-green-500 rounded-md bg-white text-green-800"
                      placeholder="Longitude"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={getLocation}
                    className="p-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                    disabled={fetchingLocation}
                  >
                    {fetchingLocation ? (
                      <Loader className="w-5 h-5 animate-spin" />
                    ) : (
                      <MapPin className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-green-800 mb-1">
                  Soil Type
                </label>
                <select
                  name="soilType"
                  value={formData.soilType}
                  onChange={handleChange}
                  className="w-full p-2 border border-green-500 rounded-md bg-white text-green-800"
                >
                  <option value="">Select Soil Type</option>
                  {soilTypes.map((soil) => (
                    <option key={soil} value={soil}>
                      {soil.charAt(0).toUpperCase() + soil.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-green-800 mb-1">
                  Previous Crop
                </label>
                <select
                  name="previousCrop"
                  value={formData.previousCrop}
                  onChange={handleChange}
                  className="w-full p-2 border border-green-500 rounded-md bg-white text-green-800"
                >
                  <option value="">Select Previous Crop</option>
                  {previousCrops.map((crop) => (
                    <option key={crop} value={crop}>
                      {crop.charAt(0).toUpperCase() + crop.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="col-span-2">
                <div className="flex items-center mb-2">
                  <input
                    type="checkbox"
                    id="useIoTData"
                    name="useIoTData"
                    checked={formData.useIoTData}
                    onChange={handleChange}
                    className="mr-2"
                  />
                  <label
                    htmlFor="useIoTData"
                    className="text-sm font-medium text-green-800"
                  >
                    Use IoT Device Data for NPK Values
                  </label>
                </div>
              </div>

              {!formData.useIoTData && (
                <>
                  <div className="col-span-2">
                    <h3 className="text-lg font-semibold text-green-800 mb-4">
                      Soil Nutrient Information
                    </h3>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-green-800 mb-1">
                      Nitrogen (N) Level: {formData.nitrogen}
                    </label>
                    <div className="flex items-center">
                      <Leaf className="w-5 h-5 text-green-600 mr-2" />
                      <input
                        type="range"
                        name="nitrogen"
                        min="0"
                        max="100"
                        value={formData.nitrogen}
                        onChange={handleChange}
                        className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-green-800 mb-1">
                      Phosphorus (P) Level: {formData.phosphorus}
                    </label>
                    <div className="flex items-center">
                      <Zap className="w-5 h-5 text-green-600 mr-2" />
                      <input
                        type="range"
                        name="phosphorus"
                        min="0"
                        max="100"
                        value={formData.phosphorus}
                        onChange={handleChange}
                        className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-green-800 mb-1">
                      Potassium (K) Level: {formData.potassium}
                    </label>
                    <div className="flex items-center">
                      <Droplets className="w-5 h-5 text-green-600 mr-2" />
                      <input
                        type="range"
                        name="potassium"
                        min="0"
                        max="100"
                        value={formData.potassium}
                        onChange={handleChange}
                        className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-green-800 mb-1">
                      pH Level: {formData.ph}
                    </label>
                    <input
                      type="range"
                      name="ph"
                      min="0"
                      max="14"
                      step="0.1"
                      value={formData.ph}
                      onChange={handleChange}
                      className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-green-800 mb-1">
                      Organic Matter (%): {formData.organicMatter}
                    </label>
                    <input
                      type="range"
                      name="organicMatter"
                      min="0"
                      max="10"
                      step="0.1"
                      value={formData.organicMatter}
                      onChange={handleChange}
                      className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-green-800 mb-4">
              What would you like to do?
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div
                className={`p-6 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedOption === "suggestions"
                    ? "border-green-600 bg-green-100"
                    : "border-green-400 bg-green-50 hover:bg-green-100"
                }`}
                onClick={() => setSelectedOption("suggestions")}
              >
                <h4 className="text-lg font-medium text-green-800 mb-2">
                  Get Crop Recommendations
                </h4>
                <p className="text-sm text-green-700 opacity-80">
                  Our AI will analyze your soil type, location, and previous
                  crop to suggest the best crops to plant.
                </p>
              </div>

              <div
                className={`p-6 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedOption === "specific"
                    ? "border-green-600 bg-green-100"
                    : "border-green-400 bg-green-50 hover:bg-green-100"
                }`}
                onClick={() => setSelectedOption("specific")}
              >
                <h4 className="text-lg font-medium text-green-800 mb-2">
                  Check Specific Crop Suitability
                </h4>
                <p className="text-sm text-green-700 opacity-80">
                  Find out if a specific crop is suitable for your field
                  conditions.
                </p>
              </div>
            </div>

            {selectedOption === "specific" && (
              <div className="mt-6">
                <label className="block text-sm font-medium text-green-800 mb-2">
                  Select Crop
                </label>
                <select
                  name="specificCrop"
                  value={formData.specificCrop}
                  onChange={handleChange}
                  className="w-full p-2 border border-green-500 rounded-md bg-white text-green-800"
                >
                  <option value="">Select Crop</option>
                  {commonCrops.map((crop) => (
                    <option key={crop} value={crop}>
                      {crop.charAt(0).toUpperCase() + crop.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {loading && <MultiStepLoader loading={loading} />}
          </div>
        );

      case 3:
        return (
          <div className="space-y-8">
            {result && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6 }}
                className="space-y-8"
              >
                {/* Main Recommendation Card */}
                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.7 }}
                  className="bg-green-100 p-6 rounded-lg border border-green-300 shadow-md"
                >
                  <h3 className="text-xl font-bold text-green-800 mb-4 flex items-center">
                    <SproutIcon className="mr-2 h-6 w-6 text-green-600" />
                    {selectedOption === "specific"
                      ? `Suitability Analysis for ${formData.specificCrop.toUpperCase()}`
                      : "Top Crop Recommendations"}
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-lg font-semibold text-green-700 mb-2">
                        Best Crops for Your Soil
                      </h4>
                      <ul className="space-y-2">
                        {result.recommendations.summary.top_5_recommended_crops.map(
                          (crop, index) => (
                            <motion.li
                              key={index}
                              initial={{ x: -20, opacity: 0 }}
                              animate={{ x: 0, opacity: 1 }}
                              transition={{ delay: index * 0.1 }}
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                              className={`flex items-center p-2 rounded-md cursor-pointer transition-colors ${
                                selectedCrop === crop
                                  ? "bg-green-200"
                                  : "bg-white/50 hover:bg-green-50"
                              }`}
                              onClick={() => toggleCropSelection(crop)}
                            >
                              <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center mr-2">
                                {index + 1}
                              </span>
                              <span className="font-medium text-green-800">
                                {crop}
                              </span>
                              {selectedCrop === crop ? (
                                <ChevronUp className="ml-auto h-5 w-5 text-green-600" />
                              ) : (
                                <ChevronDown className="ml-auto h-5 w-5 text-green-600" />
                              )}
                            </motion.li>
                          )
                        )}
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-green-700 mb-2">
                        Soil Type
                      </h4>
                      <motion.div
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                        whileHover={{ scale: 1.02 }}
                        className="bg-white p-4 rounded-lg shadow-inner text-center"
                      >
                        <p className="text-2xl font-bold text-green-800">
                          {result.soil_type}
                        </p>
                      </motion.div>
                    </div>
                  </div>
                </motion.div>

                {/* Selected Crop Details */}
                <AnimatePresence>
                  {selectedCrop && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="overflow-hidden bg-green-50 rounded-lg border border-green-200 shadow-md"
                    >
                      <div className="p-6">
                        <div className="flex items-center justify-between mb-4">
                          <h3 className="text-xl font-bold text-green-800 flex items-center">
                            <SproutIcon className="mr-2 h-6 w-6 text-green-600" />
                            {selectedCrop} Details
                          </h3>
                          <button
                            onClick={() => setSelectedCrop(null)}
                            className="text-green-600 hover:text-green-800 transition-colors"
                          >
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              width="24"
                              height="24"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <line x1="18" y1="6" x2="6" y2="18"></line>
                              <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                          </button>
                        </div>

                        {/* Planting Schedule - Always shown */}
                        <motion.div
                          className="mb-6"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.2 }}
                        >
                          <h4 className="text-lg font-semibold text-green-700 mb-2 flex items-center">
                            <CalendarClock className="w-5 h-5 text-green-600 mr-2" />
                            Planting Schedule
                          </h4>
                          <div className="p-4 bg-white rounded-lg mt-2">
                            <p className="text-green-800">
                              {
                                result.recommendations.summary
                                  .planting_schedule_recommendations[
                                  selectedCrop
                                ]
                              }
                            </p>
                          </div>
                        </motion.div>

                        {/* Irrigation - Always shown */}
                        <motion.div
                          className="mb-6"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.3 }}
                        >
                          <h4 className="text-lg font-semibold text-green-700 mb-2 flex items-center">
                            <Droplet className="w-5 h-5 text-green-600 mr-2" />
                            Irrigation Recommendations
                          </h4>
                          <div className="p-4 bg-white rounded-lg mt-2">
                            <p className="text-green-800">
                              {
                                result.recommendations.summary
                                  .Irrigation_Recommendations[selectedCrop]
                              }
                            </p>
                          </div>
                        </motion.div>

                        {/* Soil Amendments - Always shown */}
                        <motion.div
                          className="mb-6"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.4 }}
                        >
                          <h4 className="text-lg font-semibold text-green-700 mb-2 flex items-center">
                            <FlaskConical className="w-5 h-5 text-green-600 mr-2" />
                            Soil Amendments
                          </h4>
                          <div className="p-4 bg-white rounded-lg mt-2">
                            <p className="text-green-800">
                              {
                                result.recommendations.summary[
                                  "Recommended Soil Amendments and Fertilizer Adjustments"
                                ][selectedCrop]
                              }
                            </p>
                          </div>
                        </motion.div>

                        {/* Warnings - Always shown */}
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.5 }}
                        >
                          <h4 className="text-lg font-semibold text-green-700 mb-2 flex items-center">
                            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
                            Warnings & Considerations
                          </h4>
                          <div className="p-4 bg-white rounded-lg mt-2">
                            <p className="text-green-800">
                              {
                                result.recommendations.summary
                                  .warnings_and_considerations[selectedCrop]
                              }
                            </p>
                          </div>
                        </motion.div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Weather Insights */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <motion.div
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.7, delay: 0.2 }}
                    className="bg-green-50 p-6 rounded-lg border border-green-200 shadow-sm hover:shadow-md transition-shadow duration-300"
                  >
                    <h3 className="text-lg font-semibold text-green-800 mb-4 flex items-center">
                      <CloudSunIcon className="w-5 h-5 text-green-600 mr-2" />
                      Key Weather Insights
                    </h3>
                    <div className="space-y-4">
                      {[
                        {
                          icon: (
                            <Thermometer className="w-5 h-5 text-green-600 mr-2" />
                          ),
                          label: "Temperature Range",
                          value: `${result.weather_summary.min_temp}°C - ${result.weather_summary.max_temp}°C`,
                          tip: "Ideal range for most crops is between 15°C and 30°C",
                        },
                        {
                          icon: (
                            <Droplets className="w-5 h-5 text-green-600 mr-2" />
                          ),
                          label: "Total Precipitation",
                          value: `${result.weather_summary.total_precip} mm`,
                          tip: "Average annual rainfall for agriculture is typically 600-1200mm",
                        },
                        {
                          icon: (
                            <Wind className="w-5 h-5 text-green-600 mr-2" />
                          ),
                          label: "Average Wind Speed",
                          value: `${result.weather_summary.avg_wind} km/h`,
                          tip: "High winds can increase evaporation and damage plants",
                        },
                        {
                          icon: (
                            <Calendar className="w-5 h-5 text-green-600 mr-2" />
                          ),
                          label: "Forecast Period",
                          value: result.weather_summary.period,
                          tip: "Consider seasonal variations when planning",
                        },
                      ].map((item, index) => (
                        <motion.div
                          key={index}
                          initial={{ scale: 0.95, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{
                            duration: 0.5,
                            delay: 0.3 + index * 0.1,
                          }}
                          className="flex items-center group"
                        >
                          {item.icon}
                          <div>
                            <p className="text-sm text-green-700">
                              {item.label}
                            </p>
                            <p className="text-lg font-medium text-green-800">
                              {item.value}
                            </p>
                          </div>
                          <div className="ml-auto relative">
                            <Info className="w-4 h-4 text-green-400 group-hover:text-green-600 cursor-pointer" />
                            <div className="absolute hidden group-hover:block z-10 w-64 p-2 bg-white shadow-lg rounded-lg border border-green-200 text-xs text-gray-700">
                              {item.tip}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>

                  {/* Critical Insights */}
                  <motion.div
                    initial={{ x: 20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.7, delay: 0.2 }}
                    className="bg-green-50 p-6 rounded-lg border border-green-200 shadow-sm hover:shadow-md transition-shadow duration-300"
                  >
                    <h3 className="text-lg font-semibold text-green-800 mb-4 flex items-center">
                      <LightbulbIcon className="w-5 h-5 text-green-600 mr-2" />
                      Critical Insights
                    </h3>
                    <ul className="space-y-3">
                      {result.insights.map((insight, index) => (
                        <motion.li
                          key={index}
                          initial={{ x: 10, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{
                            duration: 0.5,
                            delay: 0.3 + index * 0.1,
                          }}
                          className="flex items-start p-3 rounded-md hover:bg-green-100 transition-colors duration-200"
                        >
                          <div className="mr-2 text-lg">
                            {insight.startsWith("⚠️") ? (
                              <AlertTriangle className="w-5 h-5 text-yellow-600" />
                            ) : insight.startsWith("📊") ? (
                              <Info className="w-5 h-5 text-blue-600" />
                            ) : (
                              <span>{insight.split(" ")[0]}</span>
                            )}
                          </div>
                          <span className="text-green-800">
                            {insight.split(" ").slice(1).join(" ")}
                          </span>
                        </motion.li>
                      ))}
                    </ul>
                  </motion.div>
                </div>

                {/* Soil Analysis */}
                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.7, delay: 1.1 }}
                  className="bg-green-100 p-6 rounded-lg border border-green-300 shadow-md"
                >
                  <h3 className="text-lg font-semibold text-green-800 mb-4 flex items-center">
                    <LayersIcon className="w-5 h-5 text-green-600 mr-2" />
                    Soil Analysis
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {[
                      {
                        label: "N",
                        value: result.npk_values.N,
                        ideal: "20-50",
                        unit: "ppm",
                      },
                      {
                        label: "P",
                        value: result.npk_values.P,
                        ideal: "10-30",
                        unit: "ppm",
                      },
                      {
                        label: "K",
                        value: result.npk_values.K,
                        ideal: "50-100",
                        unit: "ppm",
                      },
                      {
                        label: "pH",
                        value: result.npk_values.pH,
                        ideal: "6.0-7.0",
                        unit: "",
                      },
                      {
                        label: "Organic",
                        value: `${result.npk_values.organic_matter}`,
                        ideal: "2-5",
                        unit: "%",
                      },
                    ].map((item, index) => {
                      const numericValue = parseFloat(item.value);
                      const [idealMin, idealMax] = item.ideal
                        .split("-")
                        .map(Number);
                      const isOptimal =
                        numericValue >= idealMin && numericValue <= idealMax;

                      return (
                        <motion.div
                          key={index}
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{
                            type: "spring",
                            stiffness: 260,
                            damping: 20,
                            delay: 1.2 + index * 0.1,
                          }}
                          className={`text-center p-3 rounded-lg shadow-sm border ${
                            isOptimal
                              ? "border-green-300 bg-green-50"
                              : "border-yellow-300 bg-yellow-50"
                          }`}
                        >
                          <div className="text-sm text-green-700">
                            {item.label}
                          </div>
                          <motion.div
                            initial={{ scale: 0.5 }}
                            animate={{ scale: 1 }}
                            transition={{
                              delay: 1.4 + index * 0.1,
                              type: "spring",
                            }}
                            className="text-2xl font-bold text-green-800"
                          >
                            {item.value}
                            {item.unit}
                          </motion.div>
                          <div
                            className={`text-xs mt-1 ${
                              isOptimal ? "text-green-600" : "text-yellow-600"
                            }`}
                          >
                            Ideal: {item.ideal}
                            {item.unit}
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                </motion.div>
              </motion.div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="bg-green-50 text-green-800 min-h-screen">
      <div className="max-w-5xl mx-auto p-4 md:p-8">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-green-200">
          <div className="p-6 md:p-8">
            <div className="flex items-center justify-between mb-8">
              <h2 className="text-3xl font-bold text-green-800">
                Advanced Crop Planning System
              </h2>
              <Leaf className="w-8 h-8 text-green-600" />
            </div>

            <div className="mb-8">
              <div className="flex items-center">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex items-center">
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        step >= item
                          ? "bg-green-600 text-white"
                          : "bg-green-100 text-green-800"
                      }`}
                    >
                      {item}
                    </div>
                    {item < 3 && (
                      <div
                        className={`h-1 w-16 ${
                          step > item ? "bg-green-600" : "bg-green-100"
                        }`}
                      ></div>
                    )}
                  </div>
                ))}
              </div>
              <div className="flex justify-between mt-2 text-xs text-green-800">
                <span>Field Details</span>
                <span>Crop Selection</span>
                <span>Results</span>
              </div>
            </div>

            <form>
              {renderStep()}

              <div className="mt-8 flex justify-between">
                {step > 1 && (
                  <button
                    type="button"
                    onClick={prevStep}
                    className="px-6 py-2 bg-green-100 text-green-800 font-medium rounded-md hover:bg-green-200 transition-colors border border-green-300"
                  >
                    Back
                  </button>
                )}

                {step < 3 && (
                  <button
                    type="button"
                    onClick={nextStep}
                    className="px-6 py-2 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition-colors ml-auto"
                    disabled={loading}
                  >
                    {step === 2 && loading ? (
                      <>
                        <Loader className="inline-block w-4 h-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : step === 2 ? (
                      "Get Results"
                    ) : (
                      "Next"
                    )}
                  </button>
                )}

                {step === 3 && (
                  <button
                    type="button"
                    onClick={() => {
                      setStep(1);
                      setResult(null);
                      setSelectedOption("");
                    }}
                    className="px-6 py-2 bg-green-600 text-white font-medium rounded-md hover:bg-green-700 transition-colors ml-auto"
                  >
                    Start New Analysis
                  </button>
                )}
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
