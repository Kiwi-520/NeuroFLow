import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Camera,
  MessageSquare,
  Smile,
  Frown,
  Meh,
  Heart,
  Brain,
  Zap,
  Activity,
  AlertCircle,
  CheckCircle,
  Loader,
  Play,
  Square,
} from "lucide-react";
import toast from "react-hot-toast";

interface EmotionResult {
  predicted_emotion: string;
  confidence: number;
  emotion_probabilities: Record<string, number>;
  face_detected?: boolean;
  face_coordinates?: number[] | null;
  error?: string;
}

const SimpleEmotionAnalysis: React.FC = () => {
  const [activeMode, setActiveMode] = useState<"text" | "webcam">("text");
  const [textInput, setTextInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [textResult, setTextResult] = useState<EmotionResult | null>(null);
  const [webcamResult, setWebcamResult] = useState<EmotionResult | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Emotion colors and icons
  const emotionColors: Record<string, string> = {
    happy: "#10B981",
    sad: "#3B82F6",
    angry: "#EF4444",
    fear: "#8B5CF6",
    surprise: "#F59E0B",
    disgust: "#84CC16",
    neutral: "#6B7280",
  };

  const emotionIcons: Record<string, React.ReactNode> = {
    happy: <Smile className="w-4 h-4" />,
    sad: <Frown className="w-4 h-4" />,
    angry: <AlertCircle className="w-4 h-4" />,
    fear: <AlertCircle className="w-4 h-4" />,
    surprise: <Zap className="w-4 h-4" />,
    disgust: <Meh className="w-4 h-4" />,
    neutral: <Meh className="w-4 h-4" />,
  };

  // Analyze text emotion
  const analyzeText = async () => {
    if (!textInput.trim()) return;

    setIsAnalyzing(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/emotion/analyze-text",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: textInput }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setTextResult(result);
      toast.success("Text emotion analyzed successfully!");
    } catch (error) {
      console.error("Error analyzing text:", error);
      toast.error(
        "Failed to analyze text. Make sure the server is running on port 8000."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsWebcamActive(true);
        toast.success(
          'Webcam started! Click "Analyze Face" to detect emotions.'
        );
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
      toast.error("Failed to access webcam. Please check permissions.");
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    setWebcamResult(null);
    toast.success("Webcam stopped");
  };

  // Capture and analyze webcam frame
  const analyzeWebcamFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    context?.drawImage(videoRef.current, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL("image/jpeg", 0.8);

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append("image_data", imageData);

      const response = await fetch(
        "http://localhost:8000/api/emotion/analyze-webcam",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setWebcamResult(result);

      if (result.face_detected) {
        toast.success(`Face detected! Emotion: ${result.predicted_emotion}`);
      } else {
        toast.error(
          "No face detected. Please position your face in the camera."
        );
      }
    } catch (error) {
      console.error("Error analyzing webcam:", error);
      toast.error(
        "Failed to analyze face emotion. Make sure the server is running."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Render circular progress chart for primary emotion
  const renderCircularChart = (emotions: Record<string, number>) => {
    const primaryEmotion = Object.entries(emotions).reduce((a, b) =>
      emotions[a[0]] > emotions[b[0]] ? a : b
    );
    const percentage = (primaryEmotion[1] * 100).toFixed(1);
    const circumference = 2 * Math.PI * 45; // radius = 45
    const strokeDasharray = circumference;
    const strokeDashoffset = circumference - primaryEmotion[1] * circumference;

    return (
      <div className="flex items-center justify-center mb-6">
        <div className="relative w-36 h-36">
          <svg className="w-36 h-36 transform -rotate-90" viewBox="0 0 100 100">
            {/* Background circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke="#e5e7eb"
              strokeWidth="6"
              fill="transparent"
            />
            {/* Progress circle */}
            <circle
              cx="50"
              cy="50"
              r="45"
              stroke={emotionColors[primaryEmotion[0]] || "#6B7280"}
              strokeWidth="6"
              fill="transparent"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className="transition-all duration-2000 ease-out"
              style={{
                filter: `drop-shadow(0 0 6px ${
                  emotionColors[primaryEmotion[0]] || "#6B7280"
                }40)`,
              }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="text-3xl font-bold text-text-primary mb-1">
                {percentage}%
              </div>
              <div
                className="text-sm font-medium capitalize"
                style={{ color: emotionColors[primaryEmotion[0]] || "#6B7280" }}
              >
                {primaryEmotion[0]}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render emotion chart with enhanced bars
  const renderEmotionChart = (emotions: Record<string, number>) => {
    const sortedEmotions = Object.entries(emotions)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 7);

    return (
      <div className="space-y-4">
        {sortedEmotions.map(([emotion, score], index) => (
          <div key={emotion} className="relative">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <div
                  className="p-2 rounded-full"
                  style={{
                    backgroundColor: `${emotionColors[emotion] || "#6B7280"}20`,
                    color: emotionColors[emotion] || "#6B7280",
                  }}
                >
                  {emotionIcons[emotion] || <Brain className="w-4 h-4" />}
                </div>
                <span className="capitalize font-semibold text-text-primary">
                  {emotion}
                </span>
              </div>
              <span className="text-lg font-bold text-text-primary">
                {(score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="relative bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
              <div
                className="h-full rounded-full transition-all duration-1500 ease-out relative"
                style={{
                  width: `${score * 100}%`,
                  backgroundColor: emotionColors[emotion] || "#6B7280",
                  boxShadow: `inset 0 1px 3px rgba(0,0,0,0.2), 0 1px 2px rgba(0,0,0,0.1)`,
                  background: `linear-gradient(90deg, ${
                    emotionColors[emotion] || "#6B7280"
                  }, ${emotionColors[emotion] || "#6B7280"}dd)`,
                  animationDelay: `${index * 0.1}s`,
                }}
              >
                <div
                  className="absolute inset-0 rounded-full opacity-40"
                  style={{
                    background: `linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)`,
                    animation: score > 0.1 ? "shimmer 2s infinite" : "none",
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Get emotion intensity description
  const getEmotionIntensity = (confidence: number) => {
    if (confidence >= 0.8)
      return {
        text: "Very Strong",
        color: "text-green-600",
        bgColor: "bg-green-100",
      };
    if (confidence >= 0.6)
      return { text: "Strong", color: "text-blue-600", bgColor: "bg-blue-100" };
    if (confidence >= 0.4)
      return {
        text: "Moderate",
        color: "text-yellow-600",
        bgColor: "bg-yellow-100",
      };
    return { text: "Weak", color: "text-gray-600", bgColor: "bg-gray-100" };
  };

  // Render emotion result card
  const renderEmotionResult = (result: EmotionResult, title: string) => {
    const intensity = getEmotionIntensity(result.confidence);

    return (
      <div className="bg-surface rounded-xl p-6 border border-border shadow-lg animate-[fadeInUp_0.6s_ease-out]">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-text-primary">{title}</h3>
          <div
            className={`px-3 py-1 rounded-full ${intensity.bgColor} ${intensity.color} text-sm font-medium`}
          >
            {intensity.text}
          </div>
        </div>

        {/* Circular Progress Chart */}
        <div className="mb-6">
          {renderCircularChart(result.emotion_probabilities)}
          <div className="text-center">
            <h4 className="text-2xl font-bold capitalize text-text-primary mb-1">
              {result.predicted_emotion}
            </h4>
            <p className="text-text-secondary">Primary emotion detected</p>
            <div className="flex items-center justify-center space-x-2 mt-2">
              <Activity className="w-4 h-4 text-green-500" />
              <span className="text-sm font-medium text-green-600">
                {(result.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          </div>
        </div>

        {/* Detailed Emotion Breakdown */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-text-primary mb-4 flex items-center space-x-2">
            <Heart className="w-5 h-5 text-red-500" />
            <span>Emotion Analysis Breakdown</span>
          </h4>
          {renderEmotionChart(result.emotion_probabilities)}
        </div>

        {/* Analysis Statistics */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-background rounded-lg p-4 border border-border">
            <div className="flex items-center space-x-2 mb-2">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span className="text-sm font-medium text-text-secondary">
                Confidence
              </span>
            </div>
            <div className="text-2xl font-bold text-text-primary">
              {(result.confidence * 100).toFixed(1)}%
            </div>
          </div>
          <div className="bg-background rounded-lg p-4 border border-border">
            <div className="flex items-center space-x-2 mb-2">
              <Activity className="w-4 h-4 text-blue-500" />
              <span className="text-sm font-medium text-text-secondary">
                Emotions
              </span>
            </div>
            <div className="text-2xl font-bold text-text-primary">
              {Object.keys(result.emotion_probabilities).length}
            </div>
          </div>
        </div>

        {/* Face detection info for webcam */}
        {result.face_detected !== undefined && (
          <div className="bg-background rounded-lg p-4 border border-border mb-4">
            <div className="flex items-center space-x-2">
              {result.face_detected ? (
                <>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-green-700 font-medium">
                    Face detected successfully
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-5 h-5 text-red-500" />
                  <span className="text-red-700 font-medium">
                    No face detected in image
                  </span>
                </>
              )}
            </div>
            {result.face_coordinates && (
              <div className="text-xs text-text-secondary mt-2">
                Detection coordinates: {result.face_coordinates.join(", ")}
              </div>
            )}
          </div>
        )}

        {result.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700 font-medium">Analysis Error</span>
            </div>
            <p className="text-red-600 text-sm mt-1">{result.error}</p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="bg-primary/10 rounded-full p-4">
            <Brain className="w-8 h-8 text-primary" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-text-primary mb-2">
          Emotion Analysis
        </h1>
        <p className="text-text-secondary max-w-2xl mx-auto">
          Analyze emotions from text or detect facial emotions using your webcam
          with real-time AI processing.
        </p>
      </div>

      {/* Mode Selection */}
      <div className="flex justify-center mb-6">
        <div className="bg-surface rounded-xl p-1 border border-border">
          <button
            onClick={() => setActiveMode("text")}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all duration-200 ${
              activeMode === "text"
                ? "bg-primary text-white shadow-lg"
                : "text-text-secondary hover:text-text-primary hover:bg-background"
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span className="font-medium">Text Analysis</span>
          </button>
          <button
            onClick={() => setActiveMode("webcam")}
            className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all duration-200 ${
              activeMode === "webcam"
                ? "bg-primary text-white shadow-lg"
                : "text-text-secondary hover:text-text-primary hover:bg-background"
            }`}
          >
            <Camera className="w-4 h-4" />
            <span className="font-medium">Webcam Analysis</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <div className="bg-surface rounded-xl p-6 border border-border">
          {activeMode === "text" ? (
            <>
              <h2 className="text-xl font-semibold text-text-primary mb-4 flex items-center space-x-2">
                <MessageSquare className="w-5 h-5" />
                <span>Text Emotion Analysis</span>
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-text-primary mb-2">
                    Enter your text to analyze emotions
                  </label>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Type how you're feeling... (e.g., 'I'm so excited about this new project!')"
                    className="w-full h-32 px-4 py-3 bg-background border border-border rounded-lg 
                             text-text-primary placeholder-text-secondary resize-none
                             focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
                  />
                </div>
                <button
                  onClick={analyzeText}
                  disabled={isAnalyzing || !textInput.trim()}
                  className="w-full bg-primary hover:bg-primary/90 disabled:bg-primary/50 
                           text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200
                           flex items-center justify-center space-x-2"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="w-4 h-4 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4" />
                      <span>Analyze Text Emotion</span>
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <>
              <h2 className="text-xl font-semibold text-text-primary mb-4 flex items-center space-x-2">
                <Camera className="w-5 h-5" />
                <span>Webcam Emotion Detection</span>
              </h2>
              <div className="space-y-4">
                {/* Webcam Controls */}
                <div className="flex space-x-2">
                  {!isWebcamActive ? (
                    <button
                      onClick={startWebcam}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white font-medium 
                               py-2 px-4 rounded-lg transition-colors duration-200
                               flex items-center justify-center space-x-2"
                    >
                      <Play className="w-4 h-4" />
                      <span>Start Webcam</span>
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={analyzeWebcamFrame}
                        disabled={isAnalyzing}
                        className="flex-1 bg-primary hover:bg-primary/90 disabled:bg-primary/50 
                                 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200
                                 flex items-center justify-center space-x-2"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader className="w-4 h-4 animate-spin" />
                            <span>Analyzing...</span>
                          </>
                        ) : (
                          <>
                            <Zap className="w-4 h-4" />
                            <span>Analyze Face</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={stopWebcam}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg
                                 transition-colors duration-200 flex items-center space-x-2"
                      >
                        <Square className="w-4 h-4" />
                        <span>Stop</span>
                      </button>
                    </>
                  )}
                </div>

                {/* Video Display */}
                <div className="relative bg-background rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    className="w-full h-64 object-cover"
                    style={{ display: isWebcamActive ? "block" : "none" }}
                  />
                  {!isWebcamActive && (
                    <div className="h-64 flex items-center justify-center">
                      <div className="text-center text-text-secondary">
                        <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>
                          Click "Start Webcam" to begin face emotion detection
                        </p>
                      </div>
                    </div>
                  )}
                  <canvas ref={canvasRef} className="hidden" />
                </div>
              </div>
            </>
          )}
        </div>

        {/* Results Section */}
        <div>
          {activeMode === "text" ? (
            textResult ? (
              renderEmotionResult(textResult, "Text Emotion Analysis Results")
            ) : (
              <div className="bg-surface rounded-xl p-6 border border-border border-dashed">
                <div className="text-center text-text-secondary">
                  <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Enter text and click "Analyze" to see emotion results</p>
                </div>
              </div>
            )
          ) : webcamResult ? (
            renderEmotionResult(webcamResult, "Face Emotion Detection Results")
          ) : (
            <div className="bg-surface rounded-xl p-6 border border-border border-dashed">
              <div className="text-center text-text-secondary">
                <Camera className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Start webcam and click "Analyze Face" to detect emotions</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimpleEmotionAnalysis;
