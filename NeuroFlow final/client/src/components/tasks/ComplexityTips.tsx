import React from 'react';
import { FiTarget } from 'react-icons/fi';
import { TaskComplexity } from '../../types/task';

interface ComplexityTipsProps {
  complexity: TaskComplexity;
  tips: any;
}

const ComplexityTips: React.FC<ComplexityTipsProps> = React.memo(({ complexity, tips }) => {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div className="flex items-center space-x-2 mb-2">
        <FiTarget className="w-5 h-5 text-blue-600" />
        <h4 className="font-medium text-blue-900 capitalize">
          {complexity} Tasks
        </h4>
      </div>
      <p className="text-blue-800 text-sm mb-2">
        {tips[complexity].description}
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div>
          <span className="font-medium text-blue-900">Examples:</span>
          <ul className="text-blue-800 mt-1 space-y-1">
            {tips[complexity].examples.map((example: string, index: number) => (
              <li key={index}>• {example}</li>
            ))}
          </ul>
        </div>
        <div>
          <span className="font-medium text-blue-900">Time Range:</span>
          <p className="text-blue-800 mt-1">{tips[complexity].timeRange}</p>
        </div>
        <div>
          <span className="font-medium text-blue-900">Energy Needed:</span>
          <p className="text-blue-800 mt-1">{tips[complexity].energyNeeded}</p>
        </div>
      </div>
    </div>
  );
});

ComplexityTips.displayName = 'ComplexityTips';
export default ComplexityTips;