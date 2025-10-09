import React, { useState, useEffect } from "react";
import {
  MessageSquare,
  Brain,
  Zap,
  Activity,
  CheckCircle,
  Loader,
  TrendingUp,
  BarChart3,
  Sparkles,
  Target,
  Clock,
  Award,
} from "lucide-react";
import toast from "react-hot-toast";

interface EmotionResult {
  predicted_emotion: string;
  confidence: number;
  emotion_probabilities: Record<string, number>;
  analysis_details: {
    word_count: number;
    total_emotion_score: number;
    dominant_score: number;
    linguistic_features: {
      sentence_count: number;
      exclamation_count: number;
      question_count: number;
      capitalization_ratio: number;
      avg_word_length: number;
      emoji_count: number;
      repeated_letters: number;
      all_caps_words: number;
    };
    detailed_matches: Array<{
      word: string;
      emotion: string;
      score: number;
      type: string;
      intensity: number;
      negated: boolean;
      position: number;
    }>;
    emotions_detected: number;
  };
  processing_info: {
    processing_time_ms: number;
    algorithm_version: string;
    features_used: string[];
  };
}

const EmotionAnalysis: React.FC = () => {
  const [textInput, setTextInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<EmotionResult | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<EmotionResult[]>([]);
  const [showAdvancedDetails, setShowAdvancedDetails] = useState(false);

  // Enhanced emotion colors and icons
  const emotionConfig: Record<
    string,
    { color: string; icon: string; description: string }
  > = {
    happy: {
      color: "#10B981",
      icon: "😄",
      description: "Joy, contentment, pleasure",
    },
    sad: {
      color: "#6B7280",
      icon: "😢",
      description: "Sorrow, grief, melancholy",
    },
    angry: {
      color: "#EF4444",
      icon: "😠",
      description: "Rage, frustration, irritation",
    },
    fear: {
      color: "#8B5CF6",
      icon: "😨",
      description: "Anxiety, worry, terror",
    },
    love: {
      color: "#EC4899",
      icon: "❤️",
      description: "Affection, romance, adoration",
    },
    excitement: {
      color: "#F59E0B",
      icon: "🤩",
      description: "Enthusiasm, energy, anticipation",
    },
    surprise: {
      color: "#06B6D4",
      icon: "😲",
      description: "Astonishment, wonder, shock",
    },
    disgust: {
      color: "#84CC16",
      icon: "🤢",
      description: "Revulsion, distaste, aversion",
    },
    neutral: {
      color: "#9CA3AF",
      icon: "😐",
      description: "Balanced, calm, ordinary",
    },
  };

  const analyzeText = async () => {
    if (!textInput.trim()) {
      toast.error("Please enter some text to analyze");
      return;
    }

    setIsAnalyzing(true);

    try {
      const response = await fetch(
        "http://localhost:8001/api/emotion/analyze-text",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: textInput,
            language: "en",
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const analysisResult = await response.json();
      setResult(analysisResult);
      setAnalysisHistory((prev) => [analysisResult, ...prev.slice(0, 4)]);

      toast.success(`Emotion detected: ${analysisResult.predicted_emotion}`);
    } catch (error) {
      console.error("Error analyzing text:", error);
      toast.error(
        "Failed to analyze text. Make sure the enhanced server is running on port 8001."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearAnalysis = () => {
    setResult(null);
    setTextInput("");
    toast.success("Analysis cleared");
  };

  const getEmotionBarWidth = (probability: number): string => {
    return `${Math.max(probability * 100, 2)}%`;
  };

  // Sample texts for quick testing
  const sampleTexts = [
    "I am absolutely ecstatic and overjoyed about this incredible achievement! This is the best day of my life!",
    "I'm completely heartbroken and devastated by this terrible loss. I can't stop crying.",
    "I'm so furious and enraged about this outrageous situation! This makes my blood boil!",
    "I love you more than words can express. You are my everything and my soulmate!",
    "I'm terrified and paralyzed with fear about what might happen. This is my worst nightmare!",
    "I'm so excited and pumped up about this amazing opportunity! I can't wait to get started!",
    "What an incredible surprise! I can't believe this just happened - I'm completely shocked!",
    "This is absolutely disgusting and revolting. It makes me feel sick to my stomach.",
    "It's just a normal day, nothing particularly special happening. Everything seems ordinary.",
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Emotion Analysis
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced AI-powered text emotion analysis with context-aware
            processing and real-time insights
          </p>
        </div>

        {/* Main Analysis Section */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="xl:col-span-2 space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <div className="flex items-center space-x-3 mb-4">
                <MessageSquare className="h-6 w-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-800">
                  Text Input
                </h2>
              </div>

              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Enter your text here for advanced emotion analysis... Try expressing how you feel or describing a situation!"
                className="w-full h-32 p-4 border border-gray-200 rounded-xl resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                disabled={isAnalyzing}
              />

              <div className="flex flex-col sm:flex-row gap-3 mt-4">
                <button
                  onClick={analyzeText}
                  disabled={isAnalyzing || !textInput.trim()}
                  className={`flex-1 flex items-center justify-center space-x-2 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                    isAnalyzing || !textInput.trim()
                      ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl"
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="h-5 w-5 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-5 w-5" />
                      <span>Analyze Emotion</span>
                    </>
                  )}
                </button>

                {result && (
                  <button
                    onClick={clearAnalysis}
                    className="flex items-center justify-center space-x-2 px-6 py-3 bg-gray-100 text-gray-600 rounded-xl font-medium hover:bg-gray-200 transition-all duration-200"
                  >
                    <span>Clear</span>
                  </button>
                )}
              </div>
            </div>

            {/* Sample Texts */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Quick Test Samples
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {sampleTexts.map((sample, index) => (
                  <button
                    key={index}
                    onClick={() => setTextInput(sample)}
                    className="p-3 text-left text-sm bg-gray-50 hover:bg-blue-50 rounded-lg transition-colors duration-200 border border-gray-200 hover:border-blue-300"
                  >
                    {sample.substring(0, 50)}...
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {result && (
              <>
                {/* Primary Result */}
                <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
                  <div className="flex items-center space-x-3 mb-4">
                    <Target className="h-6 w-6 text-green-600" />
                    <h3 className="text-xl font-semibold text-gray-800">
                      Detected Emotion
                    </h3>
                  </div>

                  <div className="text-center space-y-4">
                    <div className="text-6xl">
                      {emotionConfig[result.predicted_emotion]?.icon}
                    </div>
                    <div>
                      <h4
                        className="text-2xl font-bold capitalize"
                        style={{
                          color: emotionConfig[result.predicted_emotion]?.color,
                        }}
                      >
                        {result.predicted_emotion}
                      </h4>
                      <p className="text-gray-600 text-sm mt-1">
                        {emotionConfig[result.predicted_emotion]?.description}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Processing Info */}
                <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
                  <div className="flex items-center space-x-3 mb-4">
                    <Activity className="h-6 w-6 text-blue-600" />
                    <h3 className="text-lg font-semibold text-gray-800">
                      Processing Stats
                    </h3>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center space-x-2">
                        <Clock className="h-4 w-4" />
                        <span>Processing Time</span>
                      </span>
                      <span className="font-semibold text-green-600">
                        {result.processing_info.processing_time_ms.toFixed(1)}ms
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center space-x-2">
                        <BarChart3 className="h-4 w-4" />
                        <span>Words Analyzed</span>
                      </span>
                      <span className="font-semibold text-blue-600">
                        {result.analysis_details.word_count}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center space-x-2">
                        <Award className="h-4 w-4" />
                        <span>Emotions Detected</span>
                      </span>
                      <span className="font-semibold text-purple-600">
                        {result.analysis_details.emotions_detected}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center space-x-2">
                        <TrendingUp className="h-4 w-4" />
                        <span>Algorithm</span>
                      </span>
                      <span className="font-semibold text-indigo-600">
                        v{result.processing_info.algorithm_version}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Detailed Analysis */}
        {result && (
          <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <BarChart3 className="h-6 w-6 text-purple-600" />
                <h3 className="text-xl font-semibold text-gray-800">
                  Emotion Probabilities
                </h3>
              </div>
              <button
                onClick={() => setShowAdvancedDetails(!showAdvancedDetails)}
                className="text-blue-600 hover:text-blue-700 font-medium transition-colors duration-200"
              >
                {showAdvancedDetails ? "Hide Details" : "Show Advanced Details"}
              </button>
            </div>

            <div className="space-y-3">
              {Object.entries(result.emotion_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, probability]) => (
                  <div key={emotion} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">
                          {emotionConfig[emotion]?.icon}
                        </span>
                        <span className="font-medium capitalize text-gray-800">
                          {emotion}
                        </span>
                      </div>
                      <span className="font-semibold text-gray-600">
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all duration-500"
                        style={{
                          width: getEmotionBarWidth(probability),
                          backgroundColor: emotionConfig[emotion]?.color,
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>

            {/* Advanced Details */}
            {showAdvancedDetails && (
              <div className="mt-8 pt-6 border-t border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Linguistic Features */}
                  <div>
                    <h4 className="text-lg font-semibold text-gray-800 mb-4">
                      Linguistic Features
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Sentences:</span>
                        <span className="font-medium">
                          {
                            result.analysis_details.linguistic_features
                              .sentence_count
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Exclamations:</span>
                        <span className="font-medium">
                          {
                            result.analysis_details.linguistic_features
                              .exclamation_count
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Questions:</span>
                        <span className="font-medium">
                          {
                            result.analysis_details.linguistic_features
                              .question_count
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Emojis:</span>
                        <span className="font-medium">
                          {
                            result.analysis_details.linguistic_features
                              .emoji_count
                          }
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg Word Length:</span>
                        <span className="font-medium">
                          {result.analysis_details.linguistic_features.avg_word_length.toFixed(
                            1
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">All Caps Words:</span>
                        <span className="font-medium">
                          {
                            result.analysis_details.linguistic_features
                              .all_caps_words
                          }
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Key Matches */}
                  <div>
                    <h4 className="text-lg font-semibold text-gray-800 mb-4">
                      Key Emotional Indicators
                    </h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {result.analysis_details.detailed_matches
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 10)
                        .map((match, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between p-2 bg-gray-50 rounded-lg text-sm"
                          >
                            <div className="flex items-center space-x-2">
                              <span className="font-medium text-gray-800">
                                "{match.word}"
                              </span>
                              <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded-full">
                                {match.type}
                              </span>
                              {match.negated && (
                                <span className="text-xs px-2 py-1 bg-red-100 text-red-800 rounded-full">
                                  negated
                                </span>
                              )}
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className="text-xs text-gray-500">
                                ×{match.intensity.toFixed(1)}
                              </span>
                              <span className="font-semibold text-purple-600">
                                {match.score.toFixed(1)}
                              </span>
                            </div>
                          </div>
                        ))}
                    </div>
                  </div>
                </div>

                {/* Features Used */}
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-gray-800 mb-3">
                    Analysis Features Used
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {result.processing_info.features_used.map(
                      (feature, index) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                        >
                          {feature.replace(/_/g, " ")}
                        </span>
                      )
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Analysis History */}
        {analysisHistory.length > 0 && (
          <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              Recent Analysis History
            </h3>
            <div className="space-y-3">
              {analysisHistory.map((analysis, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-xl"
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">
                      {emotionConfig[analysis.predicted_emotion]?.icon}
                    </span>
                    <div>
                      <span className="font-medium capitalize text-gray-800">
                        {analysis.predicted_emotion}
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">
                    {analysis.processing_info.processing_time_ms.toFixed(1)}ms
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmotionAnalysis;
