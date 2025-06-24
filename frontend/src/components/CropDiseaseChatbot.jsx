import { useState, useRef } from 'react';
import { 
  Upload, AlertTriangle, Camera, MessageSquare, Send, X, Info, 
  ChevronDown, ChevronUp, RefreshCw, Check, Image as ImageIcon
} from 'lucide-react';
import axios from 'axios';

// Crop Disease Detection Component
export default function CropDiseaseChatbot() {
  const [activeTab, setActiveTab] = useState('image');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'system',
      message: 'Hello! I\'m your agricultural assistant. I can help you identify crop diseases and provide solutions. What would you like to know?',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [currentMessage, setCurrentMessage] = useState('');
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);




  // Function for 
  
  // Sample disease data for demo purposes
  // const diseaseSamples = [
  //   {
  //     id: 1,
  //     name: 'Late Blight',
  //     crop: 'Potato',
  //     description: 'A serious disease of potatoes and tomatoes caused by the fungus Phytophthora infestans.',
  //     symptoms: ['Dark lesions on leaves', 'White fungal growth on undersides of leaves', 'Dark water-soaked lesions on stems'],
  //     treatment: ['Apply fungicide with chlorothalonil or copper-based formulations', 'Improve air circulation', 'Remove infected plant parts'],
  //     preventionTips: ['Use resistant varieties', 'Avoid overhead irrigation', 'Crop rotation'],
  //     severity: 'High'
  //   },
  // ];

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.match('image.*')) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle drag and drop
  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      const file = event.dataTransfer.files[0];
      if (file.type.match('image.*')) {
        setImage(file);
        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result);
        };
        reader.readAsDataURL(file);
      }
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  
  // Handle image capture (simulated)
  const handleCaptureClick = () => {
    // In a real implementation, you would use the Web Camera API
    // For this demo, we'll just trigger the file input click
    fileInputRef.current.click();
  };

  // Reset image
  const handleResetImage = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
  };
  
  // Analyze image
  const handleAnalyzeImage = async () => {
    if (!image) return;
    
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append('file', image);

    const response = await axios.post('http://127.0.0.1:5000/predict', formData, { // Replace '/upload_image' with your API endpoint
      headers: {
        'Content-Type': 'multipart/form-data', // Important for sending files
      },
    });

    const response1 = await axios.post('http://127.0.0.1:3000/storeDisease', formData, { // Replace '/upload_image' with your API endpoint
      headers: {
        'Content-Type': 'multipart/form-data', // Important for sending files
      },
    });

    setResult(response.data.prediction);
    console.log(response);
    setIsAnalyzing(false);
    
    // Simulate API call to disease detection model
    // setTimeout(() => {
    //   // For demo purposes, select a random disease from our samples
    //   const randomDisease = diseaseSamples[Math.floor(Math.random() * diseaseSamples.length)];
    //   setResult({
    //     disease: randomDisease,
    //     confidence: Math.floor(Math.random() * 30) + 70 // Random confidence between 70-99%
    //   });
    //   setIsAnalyzing(false);
    // }, 2000);
  };
  
  // Handle chat message submission
  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!currentMessage.trim()) return;
    
    // Add user message
    const userMessage = {
      role: 'user',
      message: currentMessage,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    
    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    
    // Simulate response from chatbot
    setTimeout(() => {
      let botResponse;
      
      // Simple keyword-based responses
      const lowerMessage = currentMessage.toLowerCase();
      if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        botResponse = "Hello! How can I help with your crops today?";
      } 
      else if (lowerMessage.includes('blight') || (result?.name === 'Late Blight' && lowerMessage.includes('disease'))) {
        botResponse = "Late blight affects potatoes and tomatoes. You should immediately apply copper-based fungicide, remove infected plants, and ensure good air circulation. Would you like more specific advice?";
      }
      else if (lowerMessage.includes('mildew') || (result?.name === 'Powdery Mildew' && lowerMessage.includes('disease'))) {
        botResponse = "Powdery mildew appears as white powder on leaves. Apply sulfur-based fungicide and ensure plants have proper spacing. Would you like to know about organic treatment options?";
      }
      else if (lowerMessage.includes('blast') || (result?.name === 'Rice Blast' && lowerMessage.includes('disease'))) {
        botResponse = "Rice blast is serious. Apply systemic fungicides immediately and consider draining your fields to reduce humidity. Would you like specific product recommendations?";
      }
      else if (lowerMessage.includes('treatment') || lowerMessage.includes('solution') || lowerMessage.includes('help')) {
        botResponse = "To recommend the right treatment, I need to know what crop disease you're dealing with. You can upload an image of the affected plant or describe the symptoms in detail.";
      }
      else {
        botResponse = "I'm here to help with crop disease identification and solutions. Could you provide more details about what you're observing on your plants?";
      }
      
      const systemMessage = {
        role: 'system',
        message: botResponse,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      setMessages(prev => [...prev, systemMessage]);
      
      // Scroll to bottom of chat
      if (chatEndRef.current) {
        chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-16 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="lg:text-center mb-12">
          <h2 className="text-base text-green-600 font-semibold tracking-wide uppercase">Disease Detection</h2>
          <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
            Identify and Treat Crop Diseases
          </p>
          <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
            Upload an image of your affected plant or use your camera to quickly identify diseases and get treatment recommendations.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="border-b border-gray-200">
                <nav className="-mb-px flex" aria-label="Tabs">
                  <button
                    onClick={() => setActiveTab('image')}
                    className={`${
                      activeTab === 'image'
                        ? 'border-green-500 text-green-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    } w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm focus:outline-none`}
                  >
                    <div className="flex items-center justify-center">
                      <ImageIcon className="mr-2 h-5 w-5" />
                      Image Upload
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('camera')}
                    className={`${
                      activeTab === 'camera'
                        ? 'border-green-500 text-green-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    } w-1/2 py-4 px-1 text-center border-b-2 font-medium text-sm focus:outline-none`}
                  >
                    <div className="flex items-center justify-center">
                      <Camera className="mr-2 h-5 w-5" />
                      Camera
                    </div>
                  </button>
                </nav>
              </div>

              <div className="p-6">
                {!preview ? (
                  <div 
                    className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                  >
                    {activeTab === 'image' ? (
                      <>
                        <div className="flex justify-center">
                          <Upload className="h-12 w-12 text-gray-400" />
                        </div>
                        <p className="mt-2 text-sm font-medium text-gray-900">
                          Drag and drop your image here, or
                        </p>
                        <div className="mt-2">
                          <input
                            id="file-upload"
                            name="file-upload"
                            type="file"
                            className="sr-only"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept="image/*"
                          />
                          <label
                            htmlFor="file-upload"
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                          >
                            Select Image
                          </label>
                        </div>
                        <p className="mt-2 text-xs text-gray-500">PNG, JPG, GIF up to 10MB</p>
                      </>
                    ) : (
                      <>
                        <div className="flex justify-center">
                          <Camera className="h-12 w-12 text-gray-400" />
                        </div>
                        <p className="mt-2 text-sm font-medium text-gray-900">
                          Capture an image of your affected plant
                        </p>
                        <div className="mt-2">
                          <button
                            onClick={handleCaptureClick}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                          >
                            <Camera className="mr-2 h-4 w-4" />
                            Capture Image
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="relative">
                      <img 
                        src={preview} 
                        alt="Plant sample" 
                        className="mx-auto max-h-96 rounded-lg" 
                      />
                      <button 
                        onClick={handleResetImage}
                        className="absolute top-2 right-2 bg-white rounded-full p-1 shadow-md hover:bg-gray-100"
                      >
                        <X className="h-5 w-5 text-gray-500" />
                      </button>
                    </div>
                    
                    {!result && (
                      <div className="flex justify-center">
                        <button
                          onClick={handleAnalyzeImage}
                          disabled={isAnalyzing}
                          className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                            isAnalyzing ? 'bg-gray-400' : 'bg-green-600 hover:bg-green-700'
                          } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500`}
                        >
                          {isAnalyzing ? (
                            <>
                              <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <AlertTriangle className="mr-2 h-4 w-4" />
                              Analyze for Diseases
                            </>
                          )}
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {result && (
                  <div className="mt-6 bg-white shadow overflow-hidden sm:rounded-lg">
                    <div className="px-4 py-5 sm:px-6 bg-green-50">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 bg-green-100 rounded-md p-1">
                          <Check className="h-5 w-5 text-green-500" />
                        </div>
                        <h3 className="ml-3 text-lg leading-6 font-medium text-gray-900">
                          Disease Detected: {result.name}
                        </h3>
                        <div className="ml-auto flex items-center bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
                          {result.confidence}% Confidence
                        </div>
                      </div>
                    </div>
                    <div className="border-t border-gray-200 px-4 py-5 sm:p-6">
                      <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
                        <div className="sm:col-span-1">
                          <dt className="text-sm font-medium text-gray-500">Crop</dt>
                          <dd className="mt-1 text-sm text-gray-900">{result.crop}</dd>
                        </div>
                        <div className="sm:col-span-1">
                          <dt className="text-sm font-medium text-gray-500">Severity</dt>
                          <dd className="mt-1 text-sm text-gray-900">
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              result.severity === 'High' 
                                ? 'bg-red-100 text-red-800' 
                                : result.severity === 'Medium'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {result.severity}
                            </span>
                          </dd>
                        </div>
                        <div className="sm:col-span-2">
                          <dt className="text-sm font-medium text-gray-500">Description</dt>
                          <dd className="mt-1 text-sm text-gray-900">{result.description}</dd>
                        </div>
                        <div className="sm:col-span-2">
                          <dt className="text-sm font-medium text-gray-500">Symptoms</dt>
                          <dd className="mt-1 text-sm text-gray-900">
                            <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                              {result.symptoms.map((symptom, index) => (
                                <li key={index} className="pl-3 pr-4 py-3 flex items-center text-sm">
                                  <AlertTriangle className="flex-shrink-0 h-4 w-4 text-red-500 mr-2" />
                                  {symptom}
                                </li>
                              ))}
                            </ul>
                          </dd>
                        </div>
                        <div className="sm:col-span-2">
                          <dt className="text-sm font-medium text-gray-500">Treatment</dt>
                          <dd className="mt-1 text-sm text-gray-900">
                            <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                              {result.treatment.map((treatment, index) => (
                                <li key={index} className="pl-3 pr-4 py-3 flex items-center text-sm">
                                  <Check className="flex-shrink-0 h-4 w-4 text-green-500 mr-2" />
                                  {treatment}
                                </li>
                              ))}
                            </ul>
                          </dd>
                        </div>
                        <div className="sm:col-span-2">
                          <dt className="text-sm font-medium text-gray-500">Prevention Tips</dt>
                          <dd className="mt-1 text-sm text-gray-900">
                            <ul className="border border-gray-200 rounded-md divide-y divide-gray-200">
                              {result.preventionTips.map((tip, index) => (
                                <li key={index} className="pl-3 pr-4 py-3 flex items-center text-sm">
                                  <Info className="flex-shrink-0 h-4 w-4 text-blue-500 mr-2" />
                                  {tip}
                                </li>
                              ))}
                            </ul>
                          </dd>
                        </div>
                      </dl>
                    </div>
                    <div className="bg-gray-50 px-4 py-4 sm:px-6">
                      <div className="flex items-center justify-between">
                        <button
                          onClick={() => setChatOpen(!chatOpen)}
                          className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-full shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                        >
                          <MessageSquare className="mr-1 h-4 w-4" />
                          Ask for more advice
                        </button>
                        <button
                          onClick={handleResetImage}
                          className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded-full text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                        >
                          <RefreshCw className="mr-1 h-4 w-4" />
                          Analyze Another Image
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Chat panel */}
          <div className="lg:col-span-1">
            <div className="bg-white shadow overflow-hidden sm:rounded-lg h-full flex flex-col">
              <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                <h3 className="text-lg leading-6 font-medium text-gray-900 flex items-center">
                  <MessageSquare className="mr-2 h-5 w-5 text-green-500" />
                  Agricultural Assistant
                </h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">
                  Get advice and answers to your questions
                </p>
              </div>
              <div className="flex-1 overflow-y-auto p-4 bg-gray-50 max-h-96">
                <div className="space-y-4">
                  {messages.map((msg, idx) => (
                    <div 
                      key={idx} 
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div 
                        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                          msg.role === 'user' 
                            ? 'bg-green-600 text-white' 
                            : 'bg-white text-gray-900 border border-gray-200'
                        }`}
                      >
                        <div className="text-sm">{msg.message}</div>
                        <div 
                          className={`text-xs mt-1 ${
                            msg.role === 'user' ? 'text-green-100' : 'text-gray-500'
                          }`}
                        >
                          {msg.time}
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              </div>
              <div className="p-4 border-t border-gray-200">
                <form onSubmit={handleSendMessage} className="flex">
                  <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    placeholder="Ask a question about crop diseases..."
                    className="block w-full rounded-l-md border-gray-300 shadow-sm focus:border-green-500 focus:ring-green-500 sm:text-sm"
                  />
                  <button
                    type="submit"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-r-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                  >
                    <Send className="h-4 w-4" />
                  </button>
                </form>
                <div className="mt-2 text-xs text-gray-500">
                  Ask about specific treatments, prevention methods, or get more details about the detected disease.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}