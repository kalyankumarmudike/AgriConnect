import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';
import { Layout, Button, Drawer, theme } from 'antd';
import { 
  Sprout, 
  Home, 
  Activity, 
  RotateCcw, 
  ShoppingBag, 
  Cloud, 
  Map, 
  Menu as MenuIcon, 
  X
} from 'lucide-react';

// Components
import HomePage from './components/HomePage';
import CropDiseaseChatbot from './components/CropDiseaseChatbot';
import CropRotation from './components/CropRotation';
import MarketPlace from './components/MarketPlace';
import DiseaseHeatmap from './components/DiseaseHeatmap';
import PricePrediction from './components/PricePrediction';

const { Header, Content, Footer } = Layout;

// NavLink component for consistent styling
const NavLink = ({ to, icon, label, active }) => {
  return (
    <Link to={to} className={`border-b-2 px-4 py-5 flex items-center text-sm font-medium transition-colors ${
      active 
        ? 'border-green-500 text-green-700' 
        : 'border-transparent text-gray-500 hover:text-green-600 hover:border-green-300'
    }`}>
      {icon}
      <span className="ml-2">{label}</span>
    </Link>
  );
};

// MobileNavLink component for mobile menu
const MobileNavLink = ({ to, icon, label, active, onClose }) => {
  return (
    <Link 
      to={to} 
      className={`px-4 py-3 flex items-center text-base font-medium ${
        active 
          ? 'bg-green-50 border-l-4 border-green-500 text-green-700' 
          : 'border-l-4 border-transparent text-gray-600 hover:bg-green-50 hover:border-green-300 hover:text-green-600'
      }`}
      onClick={onClose}
    >
      {icon}
      <span className="ml-3">{label}</span>
    </Link>
  );
};

function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { token } = theme.useToken();

  return (
    <Router>
      <AppContent 
        isMenuOpen={isMenuOpen} 
        setIsMenuOpen={setIsMenuOpen} 
        token={token} 
      />
    </Router>
  );
}

function AppContent({ isMenuOpen, setIsMenuOpen, token }) {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const closeMenu = () => setIsMenuOpen(false);
  
  const navItems = [
    { path: '/', label: 'Home', icon: <Home size={18} className="text-green-600" /> },
    { path: '/chatbot', label: 'Crop Doctor', icon: <Activity size={18} className="text-green-600" /> },
    { path: '/heatmap', label: 'Disease Heatmap', icon: <Map size={18} className="text-green-600" /> },
    { path: '/rotation', label: 'Crop Rotation', icon: <RotateCcw size={18} className="text-green-600" /> },
    { path: '/marketplace', label: 'Marketplace', icon: <ShoppingBag size={18} className="text-green-600" /> },
    { path: '/priceprediction', label: 'Price Prediction', icon: <ShoppingBag size={18} className="text-green-600" /> }
  ];

  return (
    <Layout className="min-h-screen">
      {/* Header */}
      <Header className={`fixed w-full z-10 transition-all px-0 ${
        scrolled ? 'bg-white/90 backdrop-blur-md shadow-md' : 'bg-white shadow-sm'
      }`} style={{ padding: 0, height: 'auto' }}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            {/* Logo and Desktop Navigation */}
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Sprout className="h-8 w-8 text-green-600" />
                <span className="ml-2 text-xl font-bold text-green-800">AgriTech</span>
              </div>
              
              {/* Desktop Navigation */}
              <div className="hidden md:ml-6 md:flex md:space-x-1">
                {navItems.map(item => (
                  <NavLink 
                    key={item.path}
                    to={item.path}
                    icon={item.icon}
                    label={item.label}
                    active={location.pathname === item.path}
                  />
                ))}
              </div>
            </div>
            
            {/* Auth Buttons */}
            <div className="hidden md:flex md:items-center md:space-x-4">
              <Button type="primary" className="bg-green-600 hover:bg-green-700 border-none">
                Login
              </Button>
              <Button className="bg-green-100 text-green-800 hover:bg-green-200 border-green-200">
                Register
              </Button>
            </div>
            
            {/* Mobile menu button */}
            <div className="flex items-center md:hidden">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="inline-flex items-center justify-center p-2 rounded-md text-gray-500 hover:text-green-600 hover:bg-green-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-green-500"
              >
                <span className="sr-only">Open main menu</span>
                {isMenuOpen ? 
                  <X className="block h-6 w-6" /> : 
                  <MenuIcon className="block h-6 w-6" />
                }
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        <Drawer
          placement="left"
          closable={false}
          onClose={closeMenu}
          open={isMenuOpen}
          width={280}
          bodyStyle={{ padding: 0 }}
          headerStyle={{ display: 'none' }}
        >
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center">
              <Sprout className="h-6 w-6 text-green-600" />
              <span className="ml-2 font-bold text-green-800">AgriTech</span>
            </div>
            <button
              onClick={closeMenu}
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="py-2">
            {navItems.map(item => (
              <MobileNavLink
                key={item.path}
                to={item.path}
                icon={item.icon}
                label={item.label}
                active={location.pathname === item.path}
                onClose={closeMenu}
              />
            ))}
          </div>
          
          <div className="p-4 border-t border-gray-200 space-y-3">
            <Button type="primary" block className="bg-green-600 hover:bg-green-700 border-none">
              Login
            </Button>
            <Button block className="bg-green-100 text-green-800 hover:bg-green-200 border-green-200">
              Register
            </Button>
          </div>
        </Drawer>
      </Header>

      {/* Main Content */}
      <Content className="pt-16">
        <div className="max-w-10xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/chatbot" element={<CropDiseaseChatbot />} />
              <Route path="/heatmap" element={<DiseaseHeatmap />} />
              <Route path="/rotation" element={<CropRotation />} />
              <Route path="/marketplace" element={<MarketPlace />} />
              <Route path="/priceprediction" element={<PricePrediction />} />
            </Routes>
          </div>
        </div>
      </Content>
    </Layout>
  );
}

export default App;