#!/usr/bin/env python3
"""
Enhanced Emotion Recognition API Test Suite
Tests the improved accuracy of the enhanced emotion recognition system
"""

import requests
import json
import time
from datetime import datetime

# Configuration for enhanced API
API_URL = "http://localhost:8001/api/emotion/analyze-text"
HEALTH_URL = "http://localhost:8001/health"

def test_enhanced_api_health():
    """Test if enhanced API is running"""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_text_enhanced(text, language="en"):
    """Analyze text emotion using enhanced API"""
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

def run_enhanced_accuracy_tests():
    """Run comprehensive accuracy tests for the enhanced API"""
    print("🚀 Enhanced Emotion Recognition API - Accuracy Test Suite")
    print("=" * 70)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check API health
    print("1. 🏥 Checking Enhanced API Health...")
    if test_enhanced_api_health():
        print("   ✅ Enhanced API is running and healthy on port 8001")
    else:
        print("   ❌ Enhanced API is not accessible")
        print("   💡 Make sure to run: python enhanced_emotion_api.py")
        return False
    
    print()
    
    # Enhanced test cases with challenging scenarios
    test_cases = [
        # Basic emotion tests
        {
            "text": "I am absolutely thrilled and overjoyed about this incredible achievement!",
            "expected": "happy",
            "description": "Strong positive emotion with superlatives"
        },
        {
            "text": "I'm completely heartbroken and devastated by this terrible loss.",
            "expected": "sad", 
            "description": "Deep sadness and grief"
        },
        {
            "text": "I am absolutely furious and enraged about this outrageous situation!",
            "expected": "angry",
            "description": "Intense anger with strong language"
        },
        {
            "text": "I'm terrified and paralyzed with fear about what might happen.",
            "expected": "fear",
            "description": "Fear and anxiety"
        },
        {
            "text": "I love you more than words can express! You're my everything!",
            "expected": "love",
            "description": "Deep romantic love"
        },
        {
            "text": "I'm so excited and pumped up about this amazing opportunity!",
            "expected": "excitement", 
            "description": "High energy and enthusiasm"
        },
        {
            "text": "What an incredible surprise! I can't believe this just happened!",
            "expected": "surprise",
            "description": "Astonishment and disbelief"
        },
        {
            "text": "This is absolutely disgusting and revolting. Makes me sick!",
            "expected": "disgust",
            "description": "Strong disgust and revulsion"
        },
        {
            "text": "The meeting is scheduled for tomorrow at 3 PM.",
            "expected": "neutral",
            "description": "Neutral factual statement"
        },
        
        # Challenging cases that should now work better
        {
            "text": "I'm tired and frustrated with work today.",
            "expected": "angry",
            "description": "Mild frustration (should detect anger)"
        },
        {
            "text": "Finally finished that project! What a relief!",
            "expected": "happy",
            "description": "Relief and accomplishment"
        },
        {
            "text": "I'm worried about the test results tomorrow.",
            "expected": "fear",
            "description": "Anxiety about future events"
        },
        {
            "text": "That movie was amazing! Loved every minute of it! 🎬✨",
            "expected": "happy",
            "description": "Positive with emojis"
        },
        {
            "text": "Ugh, this traffic is driving me crazy! 😤",
            "expected": "angry",
            "description": "Mild irritation with emoji"
        },
        {
            "text": "Can't wait for the weekend! Going to be epic! 🎉",
            "expected": "excitement",
            "description": "Anticipation with emoji"
        },
        
        # Negation tests
        {
            "text": "I'm not happy about this situation at all.",
            "expected": "angry",  # or sad - negated happiness
            "description": "Negated happiness"
        },
        {
            "text": "This is not disgusting, it's actually quite nice.",
            "expected": "happy",  # negated disgust
            "description": "Negated disgust"
        },
        
        # Complex emotional expressions
        {
            "text": "I'm so proud of my daughter's achievement! She worked so hard!",
            "expected": "happy",
            "description": "Pride and joy"
        },
        {
            "text": "Missing my family so much right now. Haven't seen them in months.",
            "expected": "sad",
            "description": "Longing and sadness"
        },
        {
            "text": "This presentation is making me so nervous. What if I mess up?",
            "expected": "fear",
            "description": "Performance anxiety"
        }
    ]
    
    # Run tests
    results = []
    total_processing_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"2.{i:2d} 🧪 Testing: {test_case['description']}")
        print(f"      Text: \"{test_case['text']}\"")
        
        result = analyze_text_enhanced(test_case["text"])
        
        if result["success"]:
            data = result["data"]
            predicted = data["predicted_emotion"]
            confidence = data["confidence"]
            processing_time = data["processing_info"]["processing_time_ms"]
            total_processing_time += processing_time
            
            # Check if prediction matches expected
            is_correct = predicted == test_case["expected"]
            
            # For complex cases, also check if it's in top 2 emotions
            emotion_probs = data["emotion_probabilities"]
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            top_2_emotions = [emotion for emotion, _ in sorted_emotions[:2]]
            in_top_2 = test_case["expected"] in top_2_emotions
            
            if is_correct:
                status = "✅ CORRECT"
                color = "green"
            elif in_top_2:
                status = "⚠️  TOP-2"
                color = "yellow"
            else:
                status = "❌ WRONG"
                color = "red"
            
            print(f"      {status}: Predicted: {predicted} (confidence: {confidence:.1%})")
            print(f"      ⚡ Processing: {processing_time:.1f}ms | Response: {result['response_time']:.1f}ms")
            
            # Show detailed analysis for interesting cases
            if data["analysis_details"]["detailed_matches"]:
                top_matches = sorted(data["analysis_details"]["detailed_matches"], 
                                   key=lambda x: x["score"], reverse=True)[:3]
                print(f"      🔍 Top indicators: {', '.join([f'{m[\"word\"]}({m[\"score\"]:.1f})' for m in top_matches])}")
            
            results.append({
                "test_case": test_case,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
                "in_top_2": in_top_2,
                "processing_time": processing_time
            })
        else:
            print(f"      ❌ Error: {result['error']}")
            results.append({
                "test_case": test_case,
                "predicted": None,
                "confidence": 0,
                "correct": False,
                "in_top_2": False,
                "processing_time": 0
            })
        
        print()
    
    # Enhanced Summary
    print("📊 ENHANCED ACCURACY TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    correct_predictions = sum(1 for r in results if r["correct"])
    top_2_predictions = sum(1 for r in results if r["in_top_2"])
    
    exact_accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    top_2_accuracy = (top_2_predictions / total_tests) * 100 if total_tests > 0 else 0
    avg_processing_time = total_processing_time / total_tests if total_tests > 0 else 0
    avg_confidence = sum(r["confidence"] for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"📈 Exact Match Accuracy: {correct_predictions}/{total_tests} ({exact_accuracy:.1f}%)")
    print(f"🎯 Top-2 Accuracy: {top_2_predictions}/{total_tests} ({top_2_accuracy:.1f}%)")
    print(f"🎯 Average Confidence: {avg_confidence:.1%}")
    print(f"⚡ Average Processing Time: {avg_processing_time:.1f}ms")
    print()
    
    # Performance assessment with enhanced criteria
    if exact_accuracy >= 85:
        print("🎉 EXCELLENT: Outstanding accuracy achieved!")
        accuracy_grade = "A+"
    elif exact_accuracy >= 75:
        print("👍 VERY GOOD: High accuracy performance")
        accuracy_grade = "A"
    elif exact_accuracy >= 65:
        print("✅ GOOD: Satisfactory accuracy")
        accuracy_grade = "B"
    elif exact_accuracy >= 55:
        print("⚠️  FAIR: Room for improvement")
        accuracy_grade = "C"
    else:
        print("❌ POOR: Significant improvements needed")
        accuracy_grade = "D"
    
    if top_2_accuracy >= 90:
        print("🏆 EXCEPTIONAL: Top-2 accuracy is outstanding!")
    
    if avg_processing_time <= 50:
        print("⚡ LIGHTNING FAST: Excellent response time!")
    elif avg_processing_time <= 100:
        print("🏃 FAST: Good response time")
    elif avg_processing_time <= 200:
        print("👌 ACCEPTABLE: Reasonable response time")
    else:
        print("🐌 SLOW: Consider optimization")
    
    print()
    print(f"📊 Overall Grade: {accuracy_grade}")
    print("✨ Enhanced accuracy testing completed!")
    
    # Show failed cases for analysis
    failed_cases = [r for r in results if not r["correct"]]
    if failed_cases:
        print(f"\n🔍 ANALYSIS OF {len(failed_cases)} INCORRECT PREDICTIONS:")
        for i, result in enumerate(failed_cases[:5], 1):  # Show top 5 failures
            tc = result["test_case"]
            print(f"{i}. Expected: {tc['expected']}, Got: {result['predicted']}")
            print(f"   Text: \"{tc['text']}\"")
            print(f"   Description: {tc['description']}")
            print()
    
    return exact_accuracy >= 70  # Return success if accuracy is 70% or higher

if __name__ == "__main__":
    success = run_enhanced_accuracy_tests()
    exit(0 if success else 1)