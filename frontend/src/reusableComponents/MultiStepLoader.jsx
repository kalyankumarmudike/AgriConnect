import { useState, useEffect } from 'react';

const MultiStepLoader = ({ loading }) => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            return 100;
          }
          return prev + 5;
        });
      }, 150);
    } else {
      setProgress(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loading]);
  
  return (
    <div className="w-full my-4">
      <div className="relative h-2 w-full bg-green-200 rounded-full overflow-hidden">
        <div 
          className="absolute top-0 left-0 h-full bg-green-500 transition-all duration-300 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="flex justify-between mt-2">
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${loading ? 'bg-green-500 animate-pulse' : 'bg-green-300'}`}></div>
          <span className="text-xs text-green-800">Preparing data</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${progress > 30 ? 'bg-green-500 animate-pulse' : 'bg-green-300'}`}></div>
          <span className="text-xs text-green-800">Analyzing soil</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${progress > 65 ? 'bg-green-500 animate-pulse' : 'bg-green-300'}`}></div>
          <span className="text-xs text-green-800">AI processing</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${progress > 90 ? 'bg-green-500 animate-pulse' : 'bg-green-300'}`}></div>
          <span className="text-xs text-green-800">Finalizing</span>
        </div>
      </div>
    </div>
  );
};
export default MultiStepLoader;