import React, { useState } from "react";
import { useEmotionAnalysis } from "../hooks/useEmotionAnalysis";
import { Brain, Heart, Send, RefreshCw } from "lucide-react";
import toast from "react-hot-toast";

const EmotionAnalysis: React.FC = () => {
  const [text, setText] = useState("");
  const { analyzeEmotion, isLoading, error } = useEmotionAnalysis();
  const [result, setResult] = useState<any>(null);
  const [showPrompt, setShowPrompt] = useState(true);

  const prompts = [
    "How was your day today?",
    "What's on your mind?",
    "How are you feeling right now?",
    "Share your thoughts with me...",
    "What emotions are you experiencing?",
  ];

  const [currentPrompt] = useState(prompts[Math.floor(Math.random() * prompts.length)]);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    try {
      setShowPrompt(false);
      console.log('Analyzing text:', text);
      const analysis = await analyzeEmotion(text);
      console.log('Analysis result:', analysis);
      setResult(analysis);
      toast.success("Analysis complete!");
    } catch (err) {
      console.error("Error analyzing emotions:", err);
      toast.error(err instanceof Error ? err.message : "Failed to analyze emotion. Please try again.");
      setShowPrompt(true);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Welcome Section */}
        <div className="bg-gradient-to-r from-green-400 to-green-500 rounded-xl shadow-lg p-8 mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <div className="bg-white rounded-full p-3">
              <Heart className="h-8 w-8 text-green-500" />
            </div>
            <h1 className="text-3xl font-bold text-white">Emotional Check-in</h1>
          </div>
          <p className="text-white text-lg opacity-90">
            Welcome to your safe space. Here you can express your thoughts and feelings freely,
            and I'll help you understand the emotions behind them.
          </p>
        </div>
        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 flex items-center">
              <Brain className="h-6 w-6 mr-2 text-green-500" />
              {currentPrompt}
            </h2>
          </div>
          <div className="space-y-4">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Express yourself freely..."
              className="w-full h-40 p-4 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-gray-700 text-lg"
              style={{ fontFamily: "Inter, sans-serif" }}
            />
            <div className="flex justify-end">
              <button
                onClick={handleAnalyze}
                disabled={isLoading || !text.trim()}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg text-lg font-medium ${isLoading || !text.trim() ? "bg-gray-200 text-gray-500 cursor-not-allowed" : "bg-green-500 text-white hover:bg-green-600"} transition-all duration-200 shadow-md hover:shadow-lg`}
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-5 w-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Send className="h-5 w-5" />
                    <span>Analyze Emotions</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
        {error && (
          <div className="mt-4 p-6 bg-red-50 border border-red-200 rounded-xl">
            <p className="text-red-700 text-lg">{error}</p>
          </div>
        )}
        {result && (
          <div className="mt-8 space-y-6">
            {/* Primary Emotion Card */}
            <div className="p-8 bg-green-50 rounded-xl shadow-md">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-800">Your Primary Emotion</h3>
                  <p className="text-gray-600">Here's what I sense in your words</p>
                </div>
                <span className="text-6xl" role="img" aria-label={result.emotion}>
                  {result.emoji}
                </span>
              </div>
              <div className="flex items-baseline space-x-2">
                <p className="text-3xl font-bold text-green-600">
                  {result.emotion.charAt(0).toUpperCase() + result.emotion.slice(1)}
                </p>
                <p className="text-lg text-green-500">
                  with {(result.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>
            </div>
            {/* Emotion Spectrum Card */}
            <div className="p-8 bg-white rounded-xl shadow-md">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Emotional Spectrum</h3>
              <div className="space-y-4">
                {Object.entries(result.emotion_spectrum).map(([emotion, probability]) => {
                  const value = Number(probability);
                  return (
                    <div key={emotion} className="flex items-center">
                      <div className="w-32">
                        <span className="text-lg font-medium text-gray-700">
                          {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                        </span>
                      </div>
                      <div className="flex-1 mx-4">
                        <div className="h-6 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-green-400 to-green-500 transition-all duration-500"
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="w-20 text-lg font-medium text-gray-600">
                        {(value * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
            {/* Action Suggestion */}
            <div className="p-6 bg-white rounded-xl shadow-md border-l-4 border-green-500">
              <p className="text-lg text-gray-700">
                Based on your emotional state, you might want to:{" "}
                {result.emotion === "joy" && "Share your positive energy with others!"}
                {result.emotion === "sadness" && "Take a moment for self-care and reach out to someone you trust."}
                {result.emotion === "anger" && "Practice deep breathing and try to identify the root cause."}
                {result.emotion === "fear" && "Remember you're safe and consider sharing your concerns with someone."}
                {result.emotion === "surprise" && "Take time to process this unexpected situation."}
                {result.emotion === "disgust" && "Distance yourself from the situation and focus on positive aspects."}
                {result.emotion === "neutral" && "Use this balanced state to plan and reflect."}
                {result.emotion === "trust" && "Build on this feeling of confidence and security."}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionAnalysis;
