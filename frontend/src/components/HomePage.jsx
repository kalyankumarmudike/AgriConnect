import { useState, useEffect } from "react";
import {
  Sun,
  Cloud,
  CloudRain,
  Menu,
  X,
  ArrowRight,
  Search,
  AlertTriangle,
  BarChart3,
  Users,
  Calendar,
  Sprout,
} from "lucide-react";
import WeatherCard from "./WeatherCard";
import farming from "../assets/farm.png";

// Main Homepage Component
export default function HomePage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [weather, setWeather] = useState({
    location: "Loading...",
    temperature: "25°C",
    condition: "Sunny",
    icon: <Sun className="text-yellow-500" />,
  });
  const [cropPrices, setCropPrices] = useState([
    { name: "Rice", price: "₹2,100/quintal", trend: "+2.5%", isUp: true },
    { name: "Wheat", price: "₹1,800/quintal", trend: "+1.2%", isUp: true },
    { name: "Cotton", price: "₹5,600/quintal", trend: "-0.8%", isUp: false },
    { name: "Sugarcane", price: "₹280/quintal", trend: "+0.5%", isUp: true },
    { name: "Maize", price: "₹1,450/quintal", trend: "-0.3%", isUp: false },
  ]);

  // Animation for the hero section
  useEffect(() => {
    const interval = setInterval(() => {
      const animations = ["plant-growth", "rain-fall", "sun-shine"];
      const randomAnimation =
        animations[Math.floor(Math.random() * animations.length)];
      const element = document.getElementById("animation-container");
      if (element) {
        element.className = "";
        void element.offsetWidth; // Trigger reflow
        element.classList.add(randomAnimation);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-green-500 to-green-700 py-12 lg:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            {/* Left Content */}
            <div>
              <h1 className="text-3xl sm:text-4xl font-extrabold text-white">
                Empowering Farmers with Technology
              </h1>
              <p className="mt-4 text-lg text-green-100">
                AgriConnect provides advanced tools for farmers to identify crop
                diseases, optimize crop rotations, predict market prices, and
                connect directly with buyers.
              </p>
              <div className="mt-8 flex">
                <a
                  href="#"
                  className="bg-white text-green-700 px-6 py-3 rounded-md font-medium text-base shadow-md hover:bg-green-50"
                >
                  Get Started
                </a>
                <a
                  href="#"
                  className="ml-4 text-green-100 border border-green-200 px-6 py-3 rounded-md font-medium text-base hover:bg-green-600"
                >
                  Learn More
                </a>
              </div>
            </div>

            {/* Right Image with Float Animation */}
            <div className="flex justify-center items-center">
              <img
                src={farming}
                alt="Farming illustration"
                className="w-full max-w-md rounded-lg animate-float"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Weather Section */}
      <WeatherCard />

      {/* Crop Prices Section */}
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-green-600 font-semibold tracking-wide uppercase">
              Market Intelligence
            </h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Today's Crop Prices
            </p>
          </div>

          <div className="mt-10">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 shadow-lg rounded-lg overflow-hidden">
                <thead className="bg-green-600">
                  <tr>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider"
                    >
                      Crop
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider"
                    >
                      Current Price
                    </th>
                    <th
                      scope="col"
                      className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider"
                    >
                      Price Trend
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {cropPrices.map((crop, index) => (
                    <tr
                      key={index}
                      className={index % 2 === 0 ? "bg-white" : "bg-gray-50"}
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {crop.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {crop.price}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            crop.isUp
                              ? "bg-green-100 text-green-800"
                              : "bg-red-100 text-red-800"
                          }`}
                        >
                          {crop.trend}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-6 text-center">
              <button className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700">
                View All Prices <ArrowRight className="ml-2 h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-green-600 font-semibold tracking-wide uppercase">
              Our Features
            </h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Smart Solutions for Modern Farming
            </p>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-xl transition-shadow duration-300">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                  <AlertTriangle className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">
                  Disease Detection
                </h3>
                <p className="mt-2 text-gray-600">
                  Identify crop diseases quickly using our AI-powered image
                  recognition system and get expert advice.
                </p>
                <a
                  href="#"
                  className="mt-4 text-green-600 hover:text-green-700 inline-flex items-center"
                >
                  Learn more <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-xl transition-shadow duration-300">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                  <Calendar className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">
                  Crop Suggestion
                </h3>
                <p className="mt-2 text-gray-600">
                  Get personalized crop rotation plans based on your soil
                  conditions, location, and previous harvests.
                </p>
                <a
                  href="#"
                  className="mt-4 text-green-600 hover:text-green-700 inline-flex items-center"
                >
                  Learn more <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-xl transition-shadow duration-300">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                  <BarChart3 className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">
                  Price Prediction
                </h3>
                <p className="mt-2 text-gray-600">
                  Make informed decisions with our AI-driven market price
                  predictions for various crops.
                </p>
                <a
                  href="#"
                  className="mt-4 text-green-600 hover:text-green-700 inline-flex items-center"
                >
                  Learn more <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:shadow-xl transition-shadow duration-300">
                <div className="bg-green-100 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                  <Users className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="text-lg font-medium text-gray-900">
                  Marketplace
                </h3>
                <p className="mt-2 text-gray-600">
                  Connect directly with wholesale buyers to sell your produce
                  without middlemen.
                </p>
                <a
                  href="#"
                  className="mt-4 text-green-600 hover:text-green-700 inline-flex items-center"
                >
                  Learn more <ArrowRight className="ml-1 h-4 w-4" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-green-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center">
                <Sprout className="h-8 w-8 text-green-300" />
                <span className="ml-2 text-xl font-bold text-white">
                  AgriConnect
                </span>
              </div>
              <p className="mt-4 text-green-200">
                Empowering farmers with technology to increase productivity and
                profitability.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white">Features</h3>
              <ul className="mt-4 space-y-2">
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Disease Detection
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Crop Suggestion
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Price Prediction
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Marketplace
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white">Resources</h3>
              <ul className="mt-4 space-y-2">
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Farming Guide
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Weather Updates
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Market Trends
                  </a>
                </li>
                <li>
                  <a href="#" className="text-green-200 hover:text-white">
                    Community
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-medium text-white">Contact</h3>
              <ul className="mt-4 space-y-2">
                <li className="text-green-200">support@agriconnect.com</li>
                <li className="text-green-200">+91 1234567890</li>
                <li className="text-green-200">AgriTech Building, Bangalore</li>
              </ul>
            </div>
          </div>
          <div className="mt-8 border-t border-green-700 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-green-200">
              © 2025 AgriConnect. All rights reserved.
            </p>
            <div className="mt-4 md:mt-0 flex space-x-6">
              <a href="#" className="text-green-200 hover:text-white">
                <span className="sr-only">Facebook</span>
                <i className="fab fa-facebook-f"></i>
              </a>
              <a href="#" className="text-green-200 hover:text-white">
                <span className="sr-only">Twitter</span>
                <i className="fab fa-twitter"></i>
              </a>
              <a href="#" className="text-green-200 hover:text-white">
                <span className="sr-only">Instagram</span>
                <i className="fab fa-instagram"></i>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
