import React, { useState, useRef, useCallback } from "react";
import {
  Camera,
  Upload,
  MessageSquare,
  Smile,
  Frown,
  Meh,
  Heart,
  Brain,
  Zap,
  Activity,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Loader,
} from "lucide-react";
import toast from "react-hot-toast";

interface EmotionResult {
  predicted_emotion: string;
  confidence: number;
  emotion_probabilities: Record<string, number>;
  intensity_analysis?: {
    intensity_score: number;
    sentiment_polarity: string;
    positive_indicators: number;
    negative_indicators: number;
  };
  face_coordinates?: [number, number, number, number] | null;
  processing_time?: number;
  text_length?: number;
  word_count?: number;
}

interface AnalysisResult {
  text_analysis?: EmotionResult;
  image_analysis?: EmotionResult[];
  text_error?: string;
  image_error?: string;
}

const EmotionAnalysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"text" | "image" | "realtime">(
    "text"
  );
  const [textInput, setTextInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [textResult, setTextResult] = useState<EmotionResult | null>(null);
  const [imageResult, setImageResult] = useState<EmotionResult[] | null>(null);
  const [realtimeResult, setRealtimeResult] = useState<AnalysisResult | null>(
    null
  );
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  // Emotion color mapping for visualization
  const emotionColors: Record<string, string> = {
    happy: "#10B981", // green
    joy: "#10B981",
    sad: "#3B82F6", // blue
    angry: "#EF4444", // red
    fear: "#8B5CF6", // purple
    surprise: "#F59E0B", // yellow
    disgust: "#84CC16", // lime
    neutral: "#6B7280", // gray
  };

  // Emotion icons
  const emotionIcons: Record<string, React.ReactNode> = {
    happy: <Smile className="w-5 h-5" />,
    joy: <Heart className="w-5 h-5" />,
    sad: <Frown className="w-5 h-5" />,
    angry: <AlertCircle className="w-5 h-5" />,
    fear: <Zap className="w-5 h-5" />,
    surprise: <Activity className="w-5 h-5" />,
    disgust: <Meh className="w-5 h-5" />,
    neutral: <Brain className="w-5 h-5" />,
  };

  const analyzeText = async () => {
    if (!textInput.trim()) {
      toast.error("Please enter some text to analyze");
      return;
    }

    setIsAnalyzing(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/emotion/analyze-text",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: textInput,
            analysis_type: "comprehensive",
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setTextResult(result);
      toast.success("Text emotion analysis completed!");
    } catch (error) {
      console.error("Error analyzing text:", error);
      toast.error("Failed to analyze text emotion. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const analyzeImage = async (file: File) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(
        "http://localhost:8000/api/emotion/analyze-image",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setImageResult(result);
      toast.success("Image emotion analysis completed!");
    } catch (error) {
      console.error("Error analyzing image:", error);
      toast.error("Failed to analyze image emotion. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target?.result as string);
        };
        reader.readAsDataURL(file);
        analyzeImage(file);
      } else {
        toast.error("Please select a valid image file");
      }
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast.error("Failed to access camera");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  const captureFrame = useCallback(async () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      context?.drawImage(videoRef.current, 0, 0);

      // Convert to base64
      const imageData = canvas.toDataURL("image/jpeg", 0.8);

      // Analyze the captured frame
      setIsAnalyzing(true);
      try {
        const formData = new FormData();
        formData.append("image_data", imageData.split(",")[1]);
        formData.append("text", textInput);

        const response = await fetch(
          "http://localhost:8000/api/emotion/realtime/analyze",
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        setRealtimeResult(result);
      } catch (error) {
        console.error("Error in realtime analysis:", error);
        toast.error("Failed to analyze emotions in real-time");
      } finally {
        setIsAnalyzing(false);
      }
    }
  }, [textInput]);

  const renderEmotionChart = (emotions: Record<string, number>) => {
    const sortedEmotions = Object.entries(emotions)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);

    return (
      <div className="space-y-3">
        {sortedEmotions.map(([emotion, score], index) => (
          <div key={emotion} className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 min-w-[120px]">
              <div style={{ color: emotionColors[emotion] || "#6B7280" }}>
                {emotionIcons[emotion] || <Brain className="w-4 h-4" />}
              </div>
              <span className="capitalize font-medium text-sm">{emotion}</span>
            </div>
            <div className="flex-1 bg-surface rounded-full h-2 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${score * 100}%`,
                  backgroundColor: emotionColors[emotion] || "#6B7280",
                }}
              />
            </div>
            <span className="text-sm font-medium text-text-secondary min-w-[50px] text-right">
              {(score * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  };

  const renderAnalysisResults = (
    result: EmotionResult,
    type: "text" | "image"
  ) => (
    <div className="bg-surface rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text-primary flex items-center space-x-2">
          {type === "text" ? (
            <>
              <MessageSquare className="w-5 h-5" />
              <span>Text Emotion Analysis</span>
            </>
          ) : (
            <>
              <Camera className="w-5 h-5" />
              <span>Image Emotion Analysis</span>
            </>
          )}
        </h3>
        <div className="flex items-center space-x-2">
          <CheckCircle className="w-5 h-5 text-green-500" />
          <span className="text-sm text-text-secondary">
            {result.confidence
              ? `${(result.confidence * 100).toFixed(1)}% confidence`
              : "Analyzed"}
          </span>
        </div>
      </div>

      {/* Primary Emotion */}
      <div className="mb-6">
        <div className="flex items-center space-x-3 mb-2">
          <div
            className="p-3 rounded-full"
            style={{
              backgroundColor: `${
                emotionColors[result.predicted_emotion] || "#6B7280"
              }20`,
            }}
          >
            <div
              style={{
                color: emotionColors[result.predicted_emotion] || "#6B7280",
              }}
            >
              {emotionIcons[result.predicted_emotion] || (
                <Brain className="w-6 h-6" />
              )}
            </div>
          </div>
          <div>
            <h4 className="text-xl font-bold capitalize text-text-primary">
              {result.predicted_emotion}
            </h4>
            <p className="text-text-secondary">Primary emotion detected</p>
          </div>
        </div>
      </div>

      {/* Emotion Breakdown */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-text-primary mb-3 flex items-center space-x-2">
          <TrendingUp className="w-4 h-4" />
          <span>Emotion Breakdown</span>
        </h4>
        {renderEmotionChart(result.emotion_probabilities)}
      </div>

      {/* Additional Analysis for Text */}
      {type === "text" && result.intensity_analysis && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-text-primary mb-3">
            Intensity Analysis
          </h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-background rounded-lg p-3">
              <p className="text-xs text-text-secondary mb-1">
                Intensity Score
              </p>
              <p className="text-lg font-bold text-text-primary">
                {(result.intensity_analysis.intensity_score * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-background rounded-lg p-3">
              <p className="text-xs text-text-secondary mb-1">Sentiment</p>
              <p className="text-lg font-bold capitalize text-text-primary">
                {result.intensity_analysis.sentiment_polarity}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Processing Stats */}
      <div className="text-xs text-text-secondary space-y-1">
        {result.processing_time && (
          <p>Processing time: {(result.processing_time * 1000).toFixed(0)}ms</p>
        )}
        {result.text_length && (
          <p>
            Text length: {result.text_length} characters ({result.word_count}{" "}
            words)
          </p>
        )}
        {result.face_coordinates && (
          <p>
            Face detected at coordinates: {result.face_coordinates.join(", ")}
          </p>
        )}
      </div>
    </div>
  );

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
          Analyze emotions from text and images using advanced deep learning
          models. Get real-time insights into emotional states with high
          accuracy.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center mb-6">
        <div className="bg-surface rounded-xl p-1 border border-border">
          {[
            {
              id: "text" as const,
              label: "Text Analysis",
              icon: MessageSquare,
            },
            { id: "image" as const, label: "Image Analysis", icon: Camera },
            { id: "realtime" as const, label: "Real-time", icon: Activity },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all duration-200 ${
                activeTab === id
                  ? "bg-primary text-white shadow-lg"
                  : "text-text-secondary hover:text-text-primary hover:bg-background"
              }`}
            >
              <Icon className="w-4 h-4" />
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content based on active tab */}
      {activeTab === "text" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Text Input */}
          <div className="bg-surface rounded-xl p-6 border border-border">
            <h2 className="text-xl font-semibold text-text-primary mb-4 flex items-center space-x-2">
              <MessageSquare className="w-5 h-5" />
              <span>Text Emotion Recognition</span>
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Enter text to analyze emotions
                </label>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Type your text here... (e.g., 'I'm feeling really excited about this new project!')"
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
                    <span>Analyze Emotion</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Text Results */}
          <div>
            {textResult ? (
              renderAnalysisResults(textResult, "text")
            ) : (
              <div className="bg-surface rounded-xl p-6 border border-border border-dashed">
                <div className="text-center text-text-secondary">
                  <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Enter text and click "Analyze Emotion" to see results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "image" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Image Input */}
          <div className="bg-surface rounded-xl p-6 border border-border">
            <h2 className="text-xl font-semibold text-text-primary mb-4 flex items-center space-x-2">
              <Camera className="w-5 h-5" />
              <span>Image Emotion Recognition</span>
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Upload an image for facial emotion analysis
                </label>
                <div
                  className="border-2 border-dashed border-border rounded-lg p-6 text-center
                           hover:border-primary/50 transition-colors duration-200 cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  {selectedImage ? (
                    <div className="space-y-2">
                      <img
                        src={selectedImage}
                        alt="Selected"
                        className="max-h-48 mx-auto rounded-lg"
                      />
                      <p className="text-sm text-text-secondary">
                        Click to change image
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Upload className="w-12 h-12 mx-auto text-text-secondary" />
                      <p className="text-text-secondary">
                        Click to upload an image or drag and drop
                      </p>
                      <p className="text-xs text-text-secondary">
                        Supports JPEG, PNG, WebP
                      </p>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </div>
            </div>
          </div>

          {/* Image Results */}
          <div>
            {imageResult && imageResult.length > 0 ? (
              <div className="space-y-4">
                {imageResult.map((result, index) => (
                  <div key={index}>
                    {renderAnalysisResults(result, "image")}
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-surface rounded-xl p-6 border border-border border-dashed">
                <div className="text-center text-text-secondary">
                  <Camera className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload an image to see emotion analysis results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === "realtime" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Real-time Input */}
          <div className="bg-surface rounded-xl p-6 border border-border">
            <h2 className="text-xl font-semibold text-text-primary mb-4 flex items-center space-x-2">
              <Activity className="w-5 h-5" />
              <span>Real-time Analysis</span>
            </h2>

            <div className="space-y-4">
              {/* Text Input for real-time */}
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Text (optional)
                </label>
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter text to analyze alongside video..."
                  className="w-full h-20 px-4 py-3 bg-background border border-border rounded-lg 
                           text-text-primary placeholder-text-secondary resize-none
                           focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary"
                />
              </div>

              {/* Camera Controls */}
              <div className="space-y-3">
                <div className="flex space-x-2">
                  {!isStreaming ? (
                    <button
                      onClick={startCamera}
                      className="flex-1 bg-primary hover:bg-primary/90 text-white font-medium 
                               py-2 px-4 rounded-lg transition-colors duration-200
                               flex items-center justify-center space-x-2"
                    >
                      <Camera className="w-4 h-4" />
                      <span>Start Camera</span>
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={captureFrame}
                        disabled={isAnalyzing}
                        className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-green-600/50 
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
                            <span>Analyze</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={stopCamera}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg
                                 transition-colors duration-200"
                      >
                        Stop
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
                    style={{ display: isStreaming ? "block" : "none" }}
                  />
                  {!isStreaming && (
                    <div className="h-64 flex items-center justify-center">
                      <div className="text-center text-text-secondary">
                        <Camera className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p>Click "Start Camera" to begin real-time analysis</p>
                      </div>
                    </div>
                  )}
                  <canvas ref={canvasRef} className="hidden" />
                </div>
              </div>
            </div>
          </div>

          {/* Real-time Results */}
          <div className="space-y-4">
            {realtimeResult ? (
              <>
                {realtimeResult.text_analysis &&
                  renderAnalysisResults(realtimeResult.text_analysis, "text")}
                {realtimeResult.image_analysis &&
                  realtimeResult.image_analysis.length > 0 &&
                  renderAnalysisResults(
                    realtimeResult.image_analysis[0],
                    "image"
                  )}
                {realtimeResult.text_error && (
                  <div
                    className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 
                                rounded-lg p-4"
                  >
                    <p className="text-red-800 dark:text-red-200">
                      Text Error: {realtimeResult.text_error}
                    </p>
                  </div>
                )}
                {realtimeResult.image_error && (
                  <div
                    className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 
                                rounded-lg p-4"
                  >
                    <p className="text-red-800 dark:text-red-200">
                      Image Error: {realtimeResult.image_error}
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-surface rounded-xl p-6 border border-border border-dashed">
                <div className="text-center text-text-secondary">
                  <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>
                    Start camera and click "Analyze" to see real-time results
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionAnalysis;
