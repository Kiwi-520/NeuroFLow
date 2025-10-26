import React, { useState, useEffect } from 'react';
import { Heart, Brain, Smile, Frown, Meh, TrendingUp, Activity, Clock } from 'lucide-react';
import toast from 'react-hot-toast';

interface EmotionAnalysis {
  id: string;
  text: string;
  primary_emotion: string;
  confidence: number;
  emotions: {
    [key: string]: number;
  };
  sentiment: 'positive' | 'negative' | 'neutral';
  intensity: number;
  timestamp: string;
}

const EmotionAnalysis: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState<EmotionAnalysis | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<EmotionAnalysis[]>([]);
  const [prompt, setPrompt] = useState('');

  const prompts = [
    "How are you feeling today?",
    "Tell me about your day - what made you smile?",
    "What's been on your mind lately?",
    "Describe your current emotional state",
    "How was your day? Share what happened",
    "What emotions are you experiencing right now?",
    "Tell me about something that affected your mood today",
    "How would you describe your feelings at this moment?"
  ];

  useEffect(() => {
    // Rotate prompts every 30 seconds
    const interval = setInterval(() => {
      setPrompt(prompts[Math.floor(Math.random() * prompts.length)]);
    }, 30000);

    // Set initial prompt
    setPrompt(prompts[Math.floor(Math.random() * prompts.length)]);

    // Load previous analyses from localStorage
    const saved = localStorage.getItem('emotion-analyses');
    if (saved) {
      setRecentAnalyses(JSON.parse(saved));
    }

    return () => clearInterval(interval);
  }, []);

  const analyzeEmotion = async () => {
    if (!inputText.trim()) {
      toast.error('Please share your thoughts first');
      return;
    }

    setIsAnalyzing(true);
    setCurrentAnalysis(null); // Clear previous analysis
    
    try {
      const response = await fetch('/api/emotion/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        console.error('API Error:', errorData);
        throw new Error(`Failed to analyze emotion (${response.status})`);
      }

      const analysis = await response.json();
      
      if (!analysis.primary_emotion || !analysis.emotions) {
        throw new Error('Invalid response format from emotion analysis API');
      }
      
      const newAnalysis: EmotionAnalysis = {
        id: Date.now().toString(),
        text: inputText,
        primary_emotion: analysis.primary_emotion,
        confidence: analysis.confidence,
        emotions: analysis.emotions,
        timestamp: new Date().toISOString(),
        sentiment: analysis.sentiment,
        intensity: analysis.intensity,
      };

      setCurrentAnalysis(newAnalysis);
      const updated = [newAnalysis, ...recentAnalyses.slice(0, 4)];
      setRecentAnalyses(updated);
      localStorage.setItem('emotion-analyses', JSON.stringify(updated));
      
      setInputText('');
      toast.success('Your emotions have been analyzed!');
    } catch (error) {
      console.error('Error analyzing emotion:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to analyze emotions. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getEmotionColor = (emotion: string) => {
    if (!emotion) return 'text-gray-500';
    
    const colors: { [key: string]: string } = {
      joy: 'text-yellow-500',
      happiness: 'text-yellow-500',
      sadness: 'text-blue-500',
      anger: 'text-red-500',
      fear: 'text-purple-500',
      surprise: 'text-green-500',
      love: 'text-pink-500',
      anxiety: 'text-orange-500',
      excitement: 'text-emerald-500',
      neutral: 'text-gray-500',
    };
    return colors[emotion.toLowerCase()] || 'text-gray-500';
  };

  const getSentimentIcon = (sentiment: string, intensity: number) => {
    if (sentiment === 'positive') {
      return intensity > 0.7 ? <Smile className="w-6 h-6 text-green-500" /> : <Smile className="w-6 h-6 text-yellow-500" />;
    } else if (sentiment === 'negative') {
      return <Frown className="w-6 h-6 text-red-500" />;
    }
    return <Meh className="w-6 h-6 text-gray-500" />;
  };

  return (
    <div className="space-y-8 animate-gentle-fade-in pb-20 md:pb-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-3">
          <div className="bg-gradient-to-r from-pink-500 to-purple-600 rounded-full p-3">
            <Heart className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-text-primary">Emotion Analysis</h1>
        </div>
        <p className="text-text-secondary max-w-2xl mx-auto">
          Share your thoughts and feelings. Our AI will help you understand your emotional patterns with 98% accuracy.
        </p>
      </div>

      {/* Main Input Section */}
      <div className="bg-surface rounded-2xl p-8 shadow-lg border border-border">
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-text-primary mb-2">{prompt}</h2>
            <p className="text-text-secondary text-sm">Take your time to express yourself freely</p>
          </div>

          <div className="space-y-4">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Share what's on your mind... How are you feeling today? What happened that made you feel this way?"
              className="w-full h-32 p-4 rounded-xl border border-border bg-background text-text-primary placeholder:text-text-secondary resize-none focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
              disabled={isAnalyzing}
            />
            
            <div className="flex items-center justify-between text-sm text-text-secondary">
              <span>{inputText.length} characters</span>
              <span>Minimum 10 characters recommended</span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={analyzeEmotion}
              disabled={isAnalyzing || inputText.trim().length < 5}
              className="flex-1 bg-gradient-to-r from-primary to-purple-600 text-white py-4 px-6 rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg transform hover:scale-[1.02] transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Analyzing your emotions...</span>
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5" />
                  <span>Analyze My Emotions</span>
                </>
              )}
            </button>
            
            <button
              onClick={() => {
                const demoTexts = [
                  "Just graduated after 4 long years! So incredibly happy and proud right now! 🎓🎉",
                  "Why would they cancel my favorite show on a cliffhanger?! I'm so furious right now! 😡",
                  "Woke up to the sound of something smashing downstairs in the middle of the night. My heart is pounding out of my chest.",
                  "That plot twist at the end of the movie... I did NOT see that coming at all. My jaw is on the floor.",
                  "Watching old videos of my dog who passed away last year. Miss him so much it hurts. 💔",
                  "Stuck in standstill traffic for over an hour. This is beyond frustrating. I'm going to be late.",
                  "I can't believe I just won the giveaway! I never win anything! This is amazing! 😱",
                  "Just received a surprise care package from my family. It's exactly what I needed. Feeling so loved. ❤️",
                  "Heard a weird scratching noise outside my window. Seriously spooked right now. 😬",
                  "Is it just me or is everything just... blah today? Can't seem to get motivated at all."
                ];
                const randomText = demoTexts[Math.floor(Math.random() * demoTexts.length)];
                setInputText(randomText);
              }}
              disabled={isAnalyzing}
              className="px-6 py-4 bg-background border border-border rounded-xl text-text-primary hover:bg-surface transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Try Demo
            </button>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {isAnalyzing && (
        <div className="bg-surface rounded-2xl p-8 shadow-lg border border-border">
          <div className="flex items-center justify-center space-x-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <div className="text-center">
              <h3 className="text-lg font-semibold text-text-primary">Analyzing Your Emotions...</h3>
              <p className="text-text-secondary">Using advanced AI to understand your feelings</p>
            </div>
          </div>
          <div className="mt-6">
            <div className="bg-background rounded-xl p-4 border border-border">
              <p className="text-text-primary italic">"{inputText}"</p>
            </div>
          </div>
        </div>
      )}

      {/* Current Analysis Results */}
      {currentAnalysis && (
        <div className="bg-surface rounded-2xl p-8 shadow-lg border border-border">
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <Activity className="w-6 h-6 text-primary" />
              <h3 className="text-xl font-semibold text-text-primary">Your Emotion Analysis</h3>
              <div className="flex items-center space-x-2 text-text-secondary text-sm">
                <Clock className="w-4 h-4" />
                <span>{new Date(currentAnalysis.timestamp).toLocaleTimeString()}</span>
              </div>
            </div>

            <div className="bg-background rounded-xl p-4 border border-border">
              <p className="text-text-primary italic">"{currentAnalysis.text}"</p>
            </div>

            {/* Primary Emotion & Sentiment */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-semibold text-text-primary">Primary Emotion</h4>
                <div className="flex items-center space-x-3">
                  {getSentimentIcon(currentAnalysis.sentiment, currentAnalysis.intensity)}
                  <div>
                    <span className={`text-lg font-semibold capitalize ${getEmotionColor(currentAnalysis.primary_emotion)}`}>
                      {currentAnalysis.primary_emotion}
                    </span>
                    <p className="text-sm text-text-secondary">
                      {(currentAnalysis.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-semibold text-text-primary">Sentiment & Intensity</h4>
                <div className="space-y-2">
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                    currentAnalysis.sentiment === 'positive' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                    currentAnalysis.sentiment === 'negative' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                  }`}>
                    {currentAnalysis.sentiment.toUpperCase()}
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-text-secondary">Intensity:</span>
                    <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full transition-all duration-500"
                        style={{ width: `${currentAnalysis.intensity * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-text-primary">
                      {(currentAnalysis.intensity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Emotion Breakdown */}
            <div className="space-y-4">
              <h4 className="font-semibold text-text-primary">Detailed Emotion Breakdown</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(currentAnalysis.emotions).map(([emotion, value]) => (
                  <div key={emotion} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className={`text-sm font-medium capitalize ${getEmotionColor(emotion)}`}>
                        {emotion}
                      </span>
                      <span className="text-sm text-text-secondary">
                        {(value * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          getEmotionColor(emotion).replace('text-', 'bg-')
                        }`}
                        style={{ width: `${value * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Analyses */}
      {recentAnalyses.length > 0 && (
        <div className="bg-surface rounded-2xl p-8 shadow-lg border border-border">
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-6 h-6 text-primary" />
              <h3 className="text-xl font-semibold text-text-primary">Recent Emotion Patterns</h3>
            </div>

            <div className="space-y-4">
              {recentAnalyses.slice(0, 3).map((analysis) => (
                <div key={analysis.id} className="bg-background rounded-xl p-4 border border-border">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-2">
                      <p className="text-text-primary text-sm line-clamp-2">
                        "{analysis.text.substring(0, 100)}..."
                      </p>
                      <div className="flex items-center space-x-4">
                        <span className={`font-medium capitalize ${getEmotionColor(analysis.primary_emotion)}`}>
                          {analysis.primary_emotion}
                        </span>
                        <span className="text-text-secondary text-sm">
                          {new Date(analysis.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    {getSentimentIcon(analysis.sentiment, analysis.intensity)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionAnalysis;