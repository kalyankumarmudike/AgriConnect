// import React, { useEffect, useState } from 'react';
// import {
//   Cloud,
//   Sun,
//   Droplets,
//   Wind,
//   ThermometerSun,
//   Loader,
//   MapPin,
//   Calendar,
//   CloudSun,
//   CloudRain,
//   CloudSnow,
//   CloudLightning,
//   CloudFog,
// } from 'lucide-react';

// const WeatherCard = ({ lat, lon }) => {
//   const [weather, setWeather] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);
//   const [animateIn, setAnimateIn] = useState(false);

//   const API_KEY = 'ce983a620087cbc73eec2e8e0012f51b';
//   const defaultLat = 28.6139;
//   const defaultLon = 77.2090;

//   const usedLat = lat || localStorage.getItem('lat') || defaultLat;
//   const usedLon = lon || localStorage.getItem('lon') || defaultLon;

//   useEffect(() => {
//     const fetchWeather = async () => {
//       try {
//         setLoading(true);
//         const response = await fetch(
//           `https://api.openweathermap.org/data/2.5/weather?lat=${usedLat}&lon=${usedLon}&appid=${API_KEY}&units=metric`
//         );
//         if (!response.ok) throw new Error('Weather data not available');
//         const data = await response.json();
//         setWeather(data);
//         localStorage.setItem('lastWeatherData', JSON.stringify(data));
//         localStorage.setItem('lastWeatherFetch', new Date().toISOString());
//         setTimeout(() => setAnimateIn(true), 100);
//       } catch (error) {
//         console.error('Weather API error:', error);
//         setError(error.message);
//         const cachedData = localStorage.getItem('lastWeatherData');
//         if (cachedData) {
//           setWeather(JSON.parse(cachedData));
//           setError('Using cached data. ' + error.message);
//           setTimeout(() => setAnimateIn(true), 100);
//         }
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchWeather();
//     const refreshInterval = setInterval(fetchWeather, 30 * 60 * 1000);
//     return () => clearInterval(refreshInterval);
//   }, [usedLat, usedLon, API_KEY]);

//   const formatDate = () => {
//     const date = new Date();
//     return date.toLocaleDateString('en-US', {
//       weekday: 'long',
//       year: 'numeric',
//       month: 'long',
//       day: 'numeric',
//     });
//   };

//   const getWeatherIcon = () => {
//     if (!weather) return <Cloud className="text-gray-400" size={48} />;
//     const id = weather.weather[0].id;
//     if (id >= 200 && id < 300) return <CloudLightning className="text-indigo-500" size={48} />;
//     if ((id >= 300 && id < 400) || (id >= 500 && id < 600)) return <CloudRain className="text-blue-500" size={48} />;
//     if (id >= 600 && id < 700) return <CloudSnow className="text-blue-400" size={48} />;
//     if (id >= 700 && id < 800) return <CloudFog className="text-gray-500" size={48} />;
//     if (id === 800) return <Sun className="text-amber-500" size={48} />;
//     return <CloudSun className="text-blue-400" size={48} />;
//   };

//   const getBackgroundStyle = () => 'bg-gradient-to-br from-white to-gray-100';

//   const LoadingSpinner = () => (
//     <div className="flex flex-col items-center justify-center h-96 w-full bg-white rounded-2xl shadow-lg">
//       <div className="animate-spin text-blue-500 mb-4">
//         <Loader size={48} />
//       </div>
//       <p className="text-gray-700 text-lg font-medium animate-pulse">
//         Loading weather data...
//       </p>
//     </div>
//   );

//   const ErrorDisplay = () => (
//     <div className="flex flex-col items-center justify-center p-8 bg-red-50 rounded-xl shadow-lg border border-red-200">
//       <Cloud className="text-red-500 mb-4" size={48} />
//       <p className="text-red-600 font-semibold mb-2">Unable to load weather data</p>
//       <p className="text-red-500 text-sm text-center">{error}</p>
//     </div>
//   );

//   if (loading) return <LoadingSpinner />;
//   if (error && !weather) return <ErrorDisplay />;
//   if (!weather) return null;

//   const {
//     name,
//     weather: [details],
//     main: { temp, feels_like, humidity, temp_min, temp_max },
//     wind: { speed },
//     sys: { country }
//   } = weather;

//   return (
//     <div className={`transition-all duration-1000 ease-in-out ${animateIn ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
//       <div className={`rounded-2xl shadow-lg overflow-hidden ${getBackgroundStyle()} w-full max-w-xl mx-auto`}>
//         <div className="p-6">
//           <div className={`transition-all duration-700 delay-100 transform ${animateIn ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'}`}>
//             <div className="flex items-center mb-1">
//               <MapPin className="text-gray-700 mr-1" size={16} />
//               <h3 className="text-gray-800 font-semibold">{name}, {country}</h3>
//             </div>
//             <p className="text-gray-600 text-sm mb-6">{formatDate()}</p>
//           </div>

//           <div className={`flex items-start justify-between transition-all duration-700 delay-200 transform ${animateIn ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'}`}>
//             <div className="flex flex-col">
//               <div className="text-5xl font-bold text-gray-800">
//                 {Math.round(temp)}
//                 <span className="text-2xl align-top ml-1">°C</span>
//               </div>
//               <p className="text-gray-600 capitalize mt-1">{details.description}</p>
//             </div>
//             <div className="animate-bounce-slow">
//               {getWeatherIcon()}
//             </div>
//           </div>

//           {/* Adjusted Cards: Width + Aspect Ratio */}
//           <div className="grid grid-cols-3 gap-3 mt-8">
//             {/* Wind Speed */}
//             <div className={`aspect-[4/3] max-w-[180px] bg-blue-50 rounded-xl p-4 transition-all duration-500 delay-300 transform ${animateIn ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'} hover:bg-blue-100 hover:scale-105 shadow`}>
//               <div className="flex items-center mb-2">
//                 <Wind className="text-blue-500 mr-2 animate-pulse" size={18} />
//                 <span className="text-gray-700 text-sm">Wind</span>
//               </div>
//               <p className="text-gray-800 text-xl font-semibold">{speed} m/s</p>
//             </div>

//             {/* Humidity */}
//             <div className={`aspect-[4/3] max-w-[180px] bg-teal-50 rounded-xl p-4 transition-all duration-500 delay-400 transform ${animateIn ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'} hover:bg-teal-100 hover:scale-105 shadow`}>
//               <div className="flex items-center mb-2">
//                 <Droplets className="text-teal-500 mr-2" size={18} />
//                 <span className="text-gray-700 text-sm">Humidity</span>
//               </div>
//               <p className="text-gray-800 text-xl font-semibold">{humidity}%</p>
//             </div>

//             {/* Min/Max Temperature */}
//             <div className={`aspect-[4/3] max-w-[180px] bg-amber-50 rounded-xl p-4 transition-all duration-500 delay-500 transform ${animateIn ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'} hover:bg-amber-100 hover:scale-105 shadow`}>
//               <div className="flex items-center mb-2">
//                 <Sun className="text-amber-500 mr-2" size={18} />
//                 <span className="text-gray-700 text-sm">Min/Max</span>
//               </div>
//               <p className="text-gray-800 text-xl font-semibold">{Math.round(temp_min)}° / {Math.round(temp_max)}°</p>
//             </div>
//           </div>

//           {/* Footer */}
//           <div className={`mt-8 flex justify-between items-center transition-all duration-500 delay-600 ${animateIn ? 'opacity-100' : 'opacity-0'}`}>
//             <div className="flex items-center">
//               <div className={`w-2 h-2 rounded-full mr-2 ${
//                 temp > 30 ? "bg-red-500" :
//                 temp > 20 ? "bg-amber-500" :
//                 temp > 10 ? "bg-green-500" : "bg-blue-500"
//               } animate-pulse`}></div>
//               <span className="text-gray-600 text-sm">
//                 {temp > 30 ? "Hot" :
//                 temp > 20 ? "Warm" :
//                 temp > 10 ? "Mild" : "Cool"}
//               </span>
//             </div>
//             <p className="text-gray-500 text-xs">
//               Updated: {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
//             </p>
//           </div>
//         </div>
//       </div>

//       {/* Custom Animation */}
//       <style jsx>{`
//         @keyframes bounce-slow {
//           0%, 100% {
//             transform: translateY(0);
//           }
//           50% {
//             transform: translateY(-10px);
//           }
//         }
//         .animate-bounce-slow {
//           animation: bounce-slow 3s ease-in-out infinite;
//         }
//       `}</style>
//     </div>
//   );
// };

// export default WeatherCard;















import React, { useEffect, useState } from 'react';
import { 
  Cloud, 
  Sun, 
  Droplets, 
  Wind, 
  ThermometerSun, 
  Loader, 
  MapPin, 
  CloudSun, 
  CloudRain,
  Snowflake,
  CloudLightning,
  Sunrise,
  Clock
} from 'lucide-react';

/**
 * WeatherCard Component - Horizontal Layout with Tailwind CSS
 * @param {number} lat - Latitude coordinate
 * @param {number} lon - Longitude coordinate
 * @returns {JSX.Element}
 */
const WeatherCard = ({ lat, lon }) => {
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_KEY = 'ce983a620087cbc73eec2e8e0012f51b';
  const defaultLat = 28.6139;  
  const defaultLon = 77.2090;

  const usedLat = lat || localStorage.getItem('lat') || defaultLat;
  const usedLon = lon || localStorage.getItem('lon') || defaultLon;

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?lat=${usedLat}&lon=${usedLon}&appid=${API_KEY}&units=metric`
        );
        if (!response.ok) throw new Error('Weather data not available');
        const data = await response.json();
        setWeather(data);
        localStorage.setItem('lastWeatherData', JSON.stringify(data));
        localStorage.setItem('lastWeatherFetch', new Date().toISOString());
      } catch (error) {
        console.error('Weather API error:', error);
        setError(error.message);
        const cachedData = localStorage.getItem('lastWeatherData');
        if (cachedData) {
          setWeather(JSON.parse(cachedData));
          setError('Using cached data. ' + error.message);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchWeather();
    const refreshInterval = setInterval(fetchWeather, 30 * 60 * 1000); // 30 minutes
    return () => clearInterval(refreshInterval);
  }, [usedLat, usedLon, API_KEY]);

  const formatDate = () => {
    const date = new Date();
    return date.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  };

  const getConditionColor = () => {
    if (!weather) return '#718096';
    const temp = weather.main.temp;
    if (temp > 30) return '#ef4444'; // Hot - red
    if (temp > 20) return '#fbbf24'; // Warm - amber
    if (temp > 10) return '#14b8a6'; // Mild - teal
    return '#3b82f6'; // Cool - blue
  };

  // Weather icon based on condition
  const getWeatherIcon = () => {
    if (!weather) return <Cloud size={64} />;
    
    const condition = weather.weather[0].main.toLowerCase();
    if (condition.includes('rain') || condition.includes('drizzle')) {
      return <CloudRain size={64} className="text-blue-500" />;
    } else if (condition.includes('cloud')) {
      return <Cloud size={64} className="text-gray-400" />;
    } else if (condition.includes('clear')) {
      return <Sun size={64} className="text-yellow-400" />;
    } else if (condition.includes('snow')) {
      return <Snowflake size={64} className="text-blue-300" />;
    } else if (condition.includes('thunder')) {
      return <CloudLightning size={64} className="text-purple-500" />;
    } else {
      return <CloudSun size={64} className="text-orange-400" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center w-full p-8 bg-white rounded-lg shadow-sm">
        <Loader size={32} className="text-green-600 animate-spin" />
        <p className="ml-3 text-lg font-medium text-gray-600">Fetching weather data...</p>
      </div>
    );
  }

  if (error && !weather) {
    return (
      <div className="w-full p-8 bg-white rounded-lg shadow-sm text-center">
        <Cloud size={48} className="mx-auto text-gray-400 mb-4" />
        <p className="text-xl font-medium text-gray-700">Unable to load weather data</p>
        <p className="text-gray-500 mt-2">{error}</p>
      </div>
    );
  }

  if (!weather) {
    return null;
  }

  const { name, weather: [details], main: { temp, feels_like, humidity }, wind: { speed }, sys: { country } } = weather;

  return (
    <div className="w-full bg-white rounded-lg shadow-md overflow-hidden mt-6">
      <div className="px-6 py-4 bg-gradient-to-r from-green-50 to-blue-50 border-b border-gray-100">
        <h2 className="flex items-center text-lg font-semibold text-gray-800">
          <CloudSun size={20} className="mr-2 text-green-600" />
          Current Weather Conditions
        </h2>
      </div>
      
      {/* Main card content */}
      <div className="p-6">
        <div className="flex flex-col lg:flex-row">
          {/* Left: Location and main weather display */}
          <div className="flex-1 lg:pr-6 lg:border-r lg:border-gray-100">
            <div className="flex items-start">
              {/* Weather icon */}
              <div className="mr-6">
                {getWeatherIcon()}
              </div>
              
              {/* Weather details */}
              <div>
                <div className="flex items-center">
                  <MapPin size={18} className="text-gray-500" />
                  <h3 className="ml-1 text-xl font-medium text-gray-800">{name}, {country}</h3>
                </div>
                <p className="text-sm text-gray-500 mb-4">{formatDate()}</p>
                
                <div className="flex items-baseline">
                  <span className="text-4xl font-bold" style={{ color: getConditionColor() }}>
                    {Math.round(temp)}
                  </span>
                  <span className="text-2xl font-medium ml-1 text-gray-600">°C</span>
                </div>
                
                <p className="text-lg capitalize text-gray-700 mt-1">
                  {details.description}
                </p>
                
                <div className="flex items-center mt-3 text-sm text-gray-600">
                  <div className="inline-flex items-center py-1 px-2 bg-gray-100 rounded-full">
                    <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: getConditionColor() }}></div>
                    <span>
                      {temp > 30 ? "Hot" : temp > 20 ? "Warm" : temp > 10 ? "Mild" : "Cool"}
                    </span>
                  </div>
                  <div className="flex items-center ml-4">
                    <Clock size={14} className="mr-1" />
                    <span>Updated: {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right: Weather attributes */}
          <div className="flex-1 mt-6 lg:mt-0 lg:pl-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Temperature Feel */}
              <div className="flex flex-col items-center p-4 bg-gradient-to-b from-orange-50 to-yellow-50 rounded-lg">
                <div className="p-3 bg-yellow-100 rounded-full mb-2">
                  <ThermometerSun size={24} className="text-yellow-600" />
                </div>
                <span className="text-2xl font-medium text-gray-800">{Math.round(feels_like)}°C</span>
                <span className="text-sm text-gray-600 mt-1">Feels Like</span>
              </div>

              {/* Wind Speed */}
              <div className="flex flex-col items-center p-4 bg-gradient-to-b from-blue-50 to-cyan-50 rounded-lg">
                <div className="p-3 bg-blue-100 rounded-full mb-2">
                  <Wind size={24} className="text-blue-600" />
                </div>
                <span className="text-2xl font-medium text-gray-800">{speed} m/s</span>
                <span className="text-sm text-gray-600 mt-1">Wind Speed</span>
              </div>

              {/* Humidity */}
              <div className="flex flex-col items-center p-4 bg-gradient-to-b from-green-50 to-teal-50 rounded-lg">
                <div className="p-3 bg-green-100 rounded-full mb-2">
                  <Droplets size={24} className="text-green-600" />
                </div>
                <span className="text-2xl font-medium text-gray-800">{humidity}%</span>
                <span className="text-sm text-gray-600 mt-1">Humidity</span>
              </div>
            </div>
            
            {/* Weather alert or farming tips based on weather */}
            <div className="mt-4 p-3 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 mb-1">Farming Tip</h4>
              <p className="text-sm text-gray-600">
                {temp > 30 
                  ? "Consider evening watering to reduce water loss from evaporation."
                  : temp > 20 
                    ? "Good conditions for most crops. Monitor soil moisture levels."
                    : temp > 10 
                      ? "Moderate temperature - ideal for many growing activities."
                      : "Protect sensitive crops from cool temperatures."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WeatherCard;