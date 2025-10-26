import React from 'react';
import { FiZap } from 'react-icons/fi';
import { EnergyLevel } from '../../types/task';

interface EnergyTipsProps {
  energyLevel: EnergyLevel;
  tips: any;
}

const EnergyTips: React.FC<EnergyTipsProps> = React.memo(({ energyLevel, tips }) => {
  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-2">
      <div className="flex items-center space-x-2 mb-1">
        <FiZap className="w-4 h-4 text-green-600" />
        <h5 className="font-medium text-green-900 text-sm capitalize">
          {energyLevel} Energy Tasks
        </h5>
      </div>
      <p className="text-green-800 text-xs mb-2">
        {tips[energyLevel].description}
      </p>
      <p className="text-green-700 text-xs">
        <span className="font-medium">Best time:</span> {tips[energyLevel].bestTime}
      </p>
    </div>
  );
});

EnergyTips.displayName = 'EnergyTips';
export default EnergyTips;