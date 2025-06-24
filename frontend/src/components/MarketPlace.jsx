import { useState } from 'react';
import { Phone, MapPin, Calendar, Search, Filter, ChevronDown, ChevronUp, MessageSquare } from 'lucide-react';

// Static data for marketplace listings
const farmersListings = [
  {
    id: 1,
    farmer: "Rajesh Kumar",
    location: "Chandigarh, Punjab",
    crop: "Wheat",
    variety: "HD-2967",
    quantity: 35,
    price: 2100,
    quality: "Premium",
    harvestDate: "2025-04-15",
    phone: "+91 98765 43210",
    description: "Fresh harvest of HD-2967 wheat with high protein content. Looking for bulk buyers."
  },
  {
    id: 2,
    farmer: "Anita Sharma",
    location: "Mysore, Karnataka",
    crop: "Rice",
    variety: "Sona Masuri",
    quantity: 28,
    price: 3600,
    quality: "Premium",
    harvestDate: "2025-04-10",
    phone: "+91 87654 32109",
    description: "Organic Sona Masuri rice harvested using traditional methods. No chemicals used."
  },
  {
    id: 3,
    farmer: "Suresh Patel",
    location: "Ahmedabad, Gujarat",
    crop: "Cotton",
    variety: "Shankar-6",
    quantity: 42,
    price: 6200,
    quality: "Standard",
    harvestDate: "2025-03-22",
    phone: "+91 76543 21098",
    description: "Medium staple cotton, good for spinning. Can provide samples upon request."
  },
  {
    id: 4,
    farmer: "Lakshmi Reddy",
    location: "Guntur, Andhra Pradesh",
    crop: "Chilli",
    variety: "Teja",
    quantity: 18,
    price: 9500,
    quality: "Premium",
    harvestDate: "2025-04-05",
    phone: "+91 65432 10987",
    description: "Spicy Teja chillies with high color value. Ideal for export quality products."
  },
  {
    id: 5,
    farmer: "Mohan Singh",
    location: "Bhatinda, Punjab",
    crop: "Maize",
    variety: "Pioneer 3377",
    quantity: 50,
    price: 1850,
    quality: "Standard",
    harvestDate: "2025-03-28",
    phone: "+91 54321 09876",
    description: "Yellow corn with good moisture content. Suitable for animal feed and processing."
  },
  {
    id: 6,
    farmer: "Priya Verma",
    location: "Nashik, Maharashtra",
    crop: "Grapes",
    variety: "Thompson Seedless",
    quantity: 12,
    price: 8500,
    quality: "Premium",
    harvestDate: "2025-04-12",
    phone: "+91 43210 98765",
    description: "Export quality seedless grapes with proper sugar content. Cold storage available."
  }
];

// Available crops for filtering
const cropOptions = ["All", "Wheat", "Rice", "Cotton", "Chilli", "Maize", "Grapes"];

// Quality options for filtering
const qualityOptions = ["All", "Standard", "Premium"];

export default function MarketPlace() {
  const [view, setView] = useState("buyer"); // "buyer" or "farmer"
  const [listings, setListings] = useState(farmersListings);
  const [searchTerm, setSearchTerm] = useState("");
  const [cropFilter, setCropFilter] = useState("All");
  const [qualityFilter, setQualityFilter] = useState("All");
  const [sortBy, setSortBy] = useState("date");
  const [sortOrder, setSortOrder] = useState("desc");
  const [showContactInfo, setShowContactInfo] = useState({});
  
  // New listing form state (for farmer view)
  const [newListing, setNewListing] = useState({
    crop: "",
    variety: "",
    quantity: "",
    price: "",
    quality: "Standard",
    harvestDate: "",
    description: ""
  });
  
  const handleSearch = (e) => {
    setSearchTerm(e.target.value);
  };
  
  const handleCropFilter = (e) => {
    setCropFilter(e.target.value);
  };
  
  const handleQualityFilter = (e) => {
    setQualityFilter(e.target.value);
  };
  
  const toggleSortOrder = () => {
    setSortOrder(sortOrder === "asc" ? "desc" : "asc");
  };
  
  const handleSortChange = (e) => {
    setSortBy(e.target.value);
  };
  
  const toggleContactInfo = (id) => {
    setShowContactInfo(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };
  
  const handleNewListingChange = (e) => {
    const { name, value } = e.target;
    setNewListing(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  const handleAddListing = () => {
    // In a real app, this would send data to backend
    alert("Your listing has been submitted and is pending review.");
    // Reset form
    setNewListing({
      crop: "",
      variety: "",
      quantity: "",
      price: "",
      quality: "Standard",
      harvestDate: "",
      description: ""
    });
  };
  
  // Filter and sort listings
  const getFilteredAndSortedListings = () => {
    return listings
      .filter(item => {
        // Search term filter
        const searchMatch = searchTerm === "" || 
          item.farmer.toLowerCase().includes(searchTerm.toLowerCase()) ||
          item.crop.toLowerCase().includes(searchTerm.toLowerCase()) ||
          item.location.toLowerCase().includes(searchTerm.toLowerCase());
        
        // Crop filter
        const cropMatch = cropFilter === "All" || item.crop === cropFilter;
        
        // Quality filter
        const qualityMatch = qualityFilter === "All" || item.quality === qualityFilter;
        
        return searchMatch && cropMatch && qualityMatch;
      })
      .sort((a, b) => {
        // Sorting
        if (sortBy === "price") {
          return sortOrder === "asc" ? a.price - b.price : b.price - a.price;
        } else if (sortBy === "date") {
          return sortOrder === "asc" 
            ? new Date(a.harvestDate) - new Date(b.harvestDate)
            : new Date(b.harvestDate) - new Date(a.harvestDate);
        }
        return 0;
      });
  };
  
  const filteredListings = getFilteredAndSortedListings();
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold text-indigo-800 mb-6">Agricultural Marketplace</h2>
      
      <div className="flex justify-between mb-6">
        <div className="flex space-x-4">
          <button 
            onClick={() => setView("buyer")}
            className={`px-4 py-2 rounded-md ${view === "buyer" 
              ? "bg-indigo-600 text-white" 
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"}`}
          >
            Browse Listings
          </button>
          <button 
            onClick={() => setView("farmer")}
            className={`px-4 py-2 rounded-md ${view === "farmer" 
              ? "bg-indigo-600 text-white" 
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"}`}
          >
            Add Your Listing
          </button>
        </div>
      </div>
      
      {view === "buyer" ? (
        <div>
          <div className="bg-gray-50 p-4 rounded-lg mb-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search crops, farmers, locations..."
                  value={searchTerm}
                  onChange={handleSearch}
                  className="pl-10 w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div className="flex items-center">
                <Filter className="mr-2 h-5 w-5 text-gray-500" />
                <select
                  value={cropFilter}
                  onChange={handleCropFilter}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  {cropOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
              
              <div className="flex items-center">
                <Filter className="mr-2 h-5 w-5 text-gray-500" />
                <select
                  value={qualityFilter}
                  onChange={handleQualityFilter}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  {qualityOptions.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="flex items-center justify-end">
              <label className="text-sm text-gray-600 mr-2">Sort by:</label>
              <select
                value={sortBy}
                onChange={handleSortChange}
                className="p-1 border border-gray-300 rounded-md text-sm mr-2"
              >
                <option value="date">Harvest Date</option>
                <option value="price">Price</option>
              </select>
              <button onClick={toggleSortOrder} className="p-1">
                {sortOrder === "asc" ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            {filteredListings.length === 0 ? (
              <div className="text-center p-8 bg-gray-50 rounded-lg">
                <p className="text-gray-500">No listings found matching your filters.</p>
              </div>
            ) : (
              filteredListings.map(listing => (
                <div key={listing.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md">
                  <div className="flex justify-between flex-wrap">
                    <div>
                      <h3 className="text-lg font-semibold text-indigo-700">{listing.crop} - {listing.variety}</h3>
                      <div className="flex items-center text-gray-600 mt-1">
                        <MapPin className="h-4 w-4 mr-1" />
                        <span className="text-sm">{listing.location}</span>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <p className="font-bold text-xl text-indigo-600">₹{listing.price}/quintal</p>
                      <span className={`inline-block px-2 py-1 rounded-full text-xs ${
                        listing.quality === "Premium" ? "bg-green-100 text-green-800" : "bg-blue-100 text-blue-800"
                      }`}>
                        {listing.quality}
                      </span>
                    </div>
                  </div>
                  
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <span className="text-gray-500">Available Quantity:</span>
                      <span className="ml-1 font-medium">{listing.quantity} quintals</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Seller:</span>
                      <span className="ml-1 font-medium">{listing.farmer}</span>
                    </div>
                    <div className="flex items-center">
                      <Calendar className="h-4 w-4 mr-1 text-gray-500" />
                      <span className="text-gray-500">Harvested:</span>
                      <span className="ml-1 font-medium">
                        {new Date(listing.harvestDate).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  
                  <p className="mt-2 text-gray-600 text-sm">{listing.description}</p>
                  
                  <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between items-center">
                    <button
                      onClick={() => toggleContactInfo(listing.id)}
                      className="flex items-center text-indigo-600 hover:text-indigo-800 text-sm font-medium"
                    >
                      {showContactInfo[listing.id] ? "Hide Contact" : "Show Contact"}
                      <ChevronDown className={`ml-1 h-4 w-4 transition-transform ${showContactInfo[listing.id] ? "rotate-180" : ""}`} />
                    </button>
                    
                    <button className="flex items-center text-green-600 hover:text-green-800 text-sm font-medium">
                      <MessageSquare className="mr-1 h-4 w-4" />
                      Message Seller
                    </button>
                  </div>
                  
                  {showContactInfo[listing.id] && (
                    <div className="mt-3 bg-indigo-50 p-3 rounded-md">
                      <div className="flex items-center">
                        <Phone className="h-4 w-4 mr-2 text-indigo-600" />
                        <span>{listing.phone}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <div className="bg-gray-50 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-indigo-700 mb-4">Add Your Crop Listing</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Crop Type</label>
              <input
                type="text"
                name="crop"
                value={newListing.crop}
                onChange={handleNewListingChange}
                placeholder="e.g. Wheat, Rice, Cotton"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Variety/Grade</label>
              <input
                type="text"
                name="variety"
                value={newListing.variety}
                onChange={handleNewListingChange}
                placeholder="e.g. HD-2967, Sona Masuri"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Available Quantity (in quintals)</label>
              <input
                type="number"
                name="quantity"
                value={newListing.quantity}
                onChange={handleNewListingChange}
                placeholder="e.g. 25"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Price per quintal (₹)</label>
              <input
                type="number"
                name="price"
                value={newListing.price}
                onChange={handleNewListingChange}
                placeholder="e.g. 2000"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Quality</label>
              <select
                name="quality"
                value={newListing.quality}
                onChange={handleNewListingChange}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="Standard">Standard</option>
                <option value="Premium">Premium</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Harvest Date</label>
              <input
                type="date"
                name="harvestDate"
                value={newListing.harvestDate}
                onChange={handleNewListingChange}
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              name="description"
              value={newListing.description}
              onChange={handleNewListingChange}
              rows="3"
              placeholder="Provide details about your crop, quality features, storage conditions, etc."
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
          
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleAddListing}
              className="px-6 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700"
            >
              Submit Listing
            </button>
          </div>
        </div>
      )}
    </div>
  );
}