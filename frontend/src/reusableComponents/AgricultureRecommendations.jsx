import { useState, useEffect } from 'react';

export default function AgricultureRecommendations({ recommendationsText }) {
  const [formattedContent, setFormattedContent] = useState('');
  
  // Function to parse markdown-style content with asterisks
  useEffect(() => {
    if (!recommendationsText) return;
    
    // Replace markdown heading patterns
    let formatted = recommendationsText
      // Format headings
      .replace(/\*\*([^*]+):\*\*/g, '<h3 class="text-xl font-bold mt-6 mb-3 text-green-700">$1</h3>')
      .replace(/\*\*([^*]+)\*\*/g, '<h4 class="text-lg font-semibold mt-4 mb-2 text-green-600">$1</h4>')
      
      // Format lists
      .replace(/^\d+\.\s+\*\*([^:]+):\*\*\s+(.*)/gm, '<div class="my-3"><span class="font-bold text-green-700">$1:</span> $2</div>')
      
      // Format italic text
      .replace(/\*([^*]+)\*/g, '<em class="text-green-600">$1</em>')
      
      // Format bullet points
      .replace(/^\s*\*\s+(.*)/gm, '<li class="ml-6 list-disc my-1">$1</li>')
      
      // Wrap lists in ul tags
      .replace(/(<li[^>]*>.*<\/li>\n)+/g, '<ul class="my-4">$&</ul>');
    
    // Format paragraphs
    formatted = formatted
      .split('\n\n')
      .map(para => {
        if (!para.trim()) return '';
        if (para.includes('<h3') || para.includes('<h4') || para.includes('<ul') || para.includes('<div')) {
          return para;
        }
        return `<p class="my-3">${para}</p>`;
      })
      .join('\n');
    
    setFormattedContent(formatted);
  }, [recommendationsText]);

  // Sample data for preview
  const sampleData = `
Detailed Recommendations
The location experiences a hot and humid climate during the specified period (May to September), with adequate rainfall concentrated in June, July, and August. The soil is loamy with a good NPK balance, and the previous crop was rice. This suggests a need for diversification and nutrient management.

**1. Top 5 Recommended Crops:**

1.  **Pigeon Pea (Arhar/Toor Dal):**  A leguminous pulse crop.
2.  **Okra (Bhindi):** A popular vegetable.
3.  **Maize (Corn):** A staple grain crop.
4.  **Sorghum (Jowar):** A drought-tolerant grain crop.
5.  **Groundnut (Peanut):** An oilseed and pulse crop.`;

  const displayContent = recommendationsText || sampleData;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-4">
      <div className="bg-green-50 p-4 rounded-md mb-6">
        <h2 className="text-2xl font-bold text-green-800 border-b-2 border-green-300 pb-2 mb-4">
          Agricultural Recommendations
        </h2>
      </div>
      
      <div 
        className="whitespace-pre-line text-green-800 prose prose-green max-w-none"
        dangerouslySetInnerHTML={{ __html: formattedContent }}
      />
    </div>
  );
}