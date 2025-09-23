import React from 'react';

const Welcome: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center">
      <div className="text-center text-white">
        <h1 className="text-4xl font-bold mb-4">Welcome to NeuroFlow</h1>
        <p className="text-xl">Your cognitive healthcare support system</p>
      </div>
    </div>
  );
};

export default Welcome;