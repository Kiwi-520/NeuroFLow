#!/usr/bin/env python3
"""
Simple test runner for Emotion Recognition API
Run this to test the emotion recognition system quickly
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/api/emotion/analyze-text"
HEALTH_URL = "http://localhost:8000/health"

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_text(text, language="en"):
    """Analyze text emotion"""
    try:
        payload = {
            "text": text,
            "language": language
        }
        
        start_time = time.time()
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "data": data,
                "response_time": (end_time - start_time) * 1000
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "response_time": (end_time - start_time) * 1000
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": 0
        }

def run_quick_tests():
    """Run quick test suite"""
    print("🚀 Emotion Recognition API - Quick Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check API health
    print("1. 🏥 Checking API Health...")
    if test_api_health():
        print("   ✅ API is running and healthy")
    else:
        print("   ❌ API is not accessible")
        print("   💡 Make sure to run: python perfect_text_emotion_api.py")
        return False
    
    print()
    
    # Test cases
    test_cases = [
        {
            "text": "I am absolutely thrilled and overjoyed about this amazing news!",
            "expected": "happy",
            "description": "Strong positive emotion"
        },
        {
            "text": "I'm completely heartbroken and devastated by this terrible loss.",
            "expected": "sad", 
            "description": "Deep sadness"
        },
        {
            "text": "I am absolutely furious about this outrageous situation!",
            "expected": "angry",
            "description": "Intense anger"
        },
        {
            "text": "I'm terrified about what might happen next.",
            "expected": "fear",
            "description": "Fear and anxiety"
        },
        {
            "text": "I love you more than words can express!",
            "expected": "love",
            "description": "Deep love"
        },
        {
            "text": "I'm so excited about this opportunity!",
            "expected": "excitement", 
            "description": "High energy"
        },
        {
            "text": "What an incredible surprise! I can't believe it!",
            "expected": "surprise",
            "description": "Astonishment"
        },
        {
            "text": "This is absolutely disgusting and revolting.",
            "expected": "disgust",
            "description": "Strong disgust"
        },
        {
            "text": "The meeting is scheduled for tomorrow at 3 PM.",
            "expected": "neutral",
            "description": "Neutral statement"
        }
    ]
    
    # Run tests
    results = []
    total_processing_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"2.{i} 🧪 Testing: {test_case['description']}")
        print(f"     Text: \"{test_case['text']}\"")
        
        result = analyze_text(test_case["text"])
        
        if result["success"]:
            data = result["data"]
            predicted = data["predicted_emotion"]
            confidence = data["confidence"]
            processing_time = data["processing_info"]["processing_time_ms"]
            total_processing_time += processing_time
            
            # Check if prediction matches expected
            is_correct = predicted == test_case["expected"]
            status = "✅" if is_correct else "⚠️"
            
            print(f"     {status} Predicted: {predicted} (confidence: {confidence:.1%})")
            print(f"     ⚡ Processing: {processing_time:.1f}ms | Response: {result['response_time']:.1f}ms")
            
            results.append({
                "test_case": test_case,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
                "processing_time": processing_time
            })
        else:
            print(f"     ❌ Error: {result['error']}")
            results.append({
                "test_case": test_case,
                "predicted": None,
                "confidence": 0,
                "correct": False,
                "processing_time": 0
            })
        
        print()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    avg_processing_time = total_processing_time / total_tests if total_tests > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"📈 Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    print(f"🎯 Average Confidence: {avg_confidence:.1%}")
    print(f"⚡ Average Processing Time: {avg_processing_time:.1f}ms")
    print()
    
    # Performance assessment
    if accuracy >= 80:
        print("🎉 EXCELLENT: High accuracy achieved!")
    elif accuracy >= 70:
        print("👍 GOOD: Satisfactory accuracy")
    elif accuracy >= 60:
        print("⚠️  FAIR: Room for improvement")
    else:
        print("❌ POOR: Significant improvements needed")
    
    if avg_processing_time <= 100:
        print("⚡ FAST: Excellent response time!")
    elif avg_processing_time <= 200:
        print("🏃 GOOD: Good response time")
    else:
        print("🐌 SLOW: Consider optimization")
    
    print()
    print("✨ Testing completed!")
    return True

def interactive_test():
    """Interactive testing mode"""
    print("🎯 Interactive Emotion Recognition Test")
    print("=" * 40)
    print("Type 'quit' to exit")
    print()
    
    if not test_api_health():
        print("❌ API is not accessible. Make sure the server is running.")
        return
    
    while True:
        text = input("💬 Enter text to analyze: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            print("   Please enter some text.\n")
            continue
        
        print("   🔄 Analyzing...")
        result = analyze_text(text)
        
        if result["success"]:
            data = result["data"]
            
            print(f"   🎯 Emotion: {data['predicted_emotion'].upper()}")
            print(f"   📊 Confidence: {data['confidence']:.1%}")
            print(f"   ⚡ Processing: {data['processing_info']['processing_time_ms']:.1f}ms")
            
            # Show top 3 emotions
            emotions = sorted(data['emotion_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
            print(f"   📋 Top emotions:")
            for emotion, prob in emotions:
                print(f"      {emotion}: {prob:.1%}")
        else:
            print(f"   ❌ Error: {result['error']}")
        
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        run_quick_tests()