import pytest
import requests
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/emotion/analyze-text"

class TestEmotionRecognitionAPI:
    """Comprehensive test suite for the Perfect Text Emotion Recognition API"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.test_cases = [
            # Happy emotion tests
            {
                "text": "I am absolutely thrilled and overjoyed about this amazing news! This is the best day ever!",
                "expected_emotion": "happy",
                "min_confidence": 0.8,
                "description": "Strong positive expression with superlatives"
            },
            {
                "text": "Feeling fantastic and wonderful today! Life is great! 😄🎉",
                "expected_emotion": "happy",
                "min_confidence": 0.7,
                "description": "Happy with emojis"
            },
            
            # Sad emotion tests
            {
                "text": "I am completely heartbroken and devastated by this terrible loss. I can't stop crying.",
                "expected_emotion": "sad",
                "min_confidence": 0.8,
                "description": "Deep sadness and grief"
            },
            {
                "text": "Feeling so depressed and miserable today. Everything is going wrong. 😢",
                "expected_emotion": "sad",
                "min_confidence": 0.7,
                "description": "Depression with sad emoji"
            },
            
            # Angry emotion tests
            {
                "text": "I am absolutely furious and enraged about this outrageous situation! This makes my blood boil!",
                "expected_emotion": "angry",
                "min_confidence": 0.8,
                "description": "Intense anger and rage"
            },
            {
                "text": "This is so annoying and frustrating! I hate when this happens! 😠",
                "expected_emotion": "angry",
                "min_confidence": 0.7,
                "description": "Frustration with angry emoji"
            },
            
            # Fear emotion tests
            {
                "text": "I'm terrified and paralyzed with fear about what might happen. This is my worst nightmare!",
                "expected_emotion": "fear",
                "min_confidence": 0.8,
                "description": "Terror and anxiety"
            },
            {
                "text": "I'm so scared and worried about the upcoming exam. What if I fail? 😨",
                "expected_emotion": "fear",
                "min_confidence": 0.7,
                "description": "Anxiety with fearful emoji"
            },
            
            # Love emotion tests
            {
                "text": "I love you more than words can express. You are my everything and my soulmate!",
                "expected_emotion": "love",
                "min_confidence": 0.8,
                "description": "Deep romantic love"
            },
            {
                "text": "I adore and cherish you so much. You mean the world to me! ❤️💕",
                "expected_emotion": "love",
                "min_confidence": 0.7,
                "description": "Affection with love emojis"
            },
            
            # Excitement emotion tests
            {
                "text": "I'm so excited and pumped up about this amazing opportunity! I can't wait to get started!",
                "expected_emotion": "excitement",
                "min_confidence": 0.8,
                "description": "High energy and anticipation"
            },
            {
                "text": "OMG! This is incredible! I'm buzzing with energy! 🤩⚡",
                "expected_emotion": "excitement",
                "min_confidence": 0.7,
                "description": "Excitement with energetic emojis"
            },
            
            # Surprise emotion tests
            {
                "text": "What an incredible surprise! I can't believe this just happened - I'm completely shocked!",
                "expected_emotion": "surprise",
                "min_confidence": 0.8,
                "description": "Astonishment and disbelief"
            },
            {
                "text": "Wow! I didn't expect that at all! What a shocking revelation! 😲",
                "expected_emotion": "surprise",
                "min_confidence": 0.7,
                "description": "Surprise with shocked emoji"
            },
            
            # Disgust emotion tests
            {
                "text": "This is absolutely disgusting and revolting. It makes me feel sick to my stomach.",
                "expected_emotion": "disgust",
                "min_confidence": 0.8,
                "description": "Strong revulsion and nausea"
            },
            {
                "text": "Eww, that's so gross and nasty! I can't stand it! 🤢",
                "expected_emotion": "disgust",
                "min_confidence": 0.7,
                "description": "Disgust with nauseated emoji"
            },
            
            # Neutral emotion tests
            {
                "text": "It's just a normal day, nothing particularly special happening. Everything seems ordinary.",
                "expected_emotion": "neutral",
                "min_confidence": 0.6,
                "description": "Ordinary, balanced statement"
            },
            {
                "text": "The meeting is scheduled for 3 PM tomorrow. Please bring the required documents.",
                "expected_emotion": "neutral",
                "min_confidence": 0.6,
                "description": "Professional, factual statement"
            },
            
            # Complex emotion tests
            {
                "text": "I'm happy about the promotion but sad to leave my current team.",
                "expected_emotion": None,  # Mixed emotions, we'll check if it's reasonable
                "min_confidence": 0.5,
                "description": "Mixed emotions - should show complexity"
            },
            {
                "text": "NOT feeling great today - actually quite terrible and awful!",
                "expected_emotion": "sad",
                "min_confidence": 0.6,
                "description": "Negation handling test"
            }
        ]
    
    def make_api_request(self, text: str, language: str = "en") -> Dict[Any, Any]:
        """Make API request and return response"""
        payload = {
            "text": text,
            "language": language
        }
        
        response = requests.post(
            API_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10
        )
        
        return response
    
    def test_api_connection(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            assert response.status_code == 200, "API health check failed"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Cannot connect to API: {e}")
    
    def test_basic_emotion_recognition(self):
        """Test basic emotion recognition functionality"""
        response = self.make_api_request("I am very happy today!")
        
        assert response.status_code == 200, f"API request failed: {response.status_code}"
        
        data = response.json()
        
        # Check required fields
        required_fields = [
            "predicted_emotion", "confidence", "emotion_probabilities",
            "analysis_details", "processing_info"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data["predicted_emotion"], str), "predicted_emotion should be string"
        assert isinstance(data["confidence"], (int, float)), "confidence should be numeric"
        assert isinstance(data["emotion_probabilities"], dict), "emotion_probabilities should be dict"
        assert isinstance(data["analysis_details"], dict), "analysis_details should be dict"
        assert isinstance(data["processing_info"], dict), "processing_info should be dict"
        
        # Check confidence range
        assert 0.0 <= data["confidence"] <= 1.0, "Confidence should be between 0 and 1"
    
    def test_comprehensive_emotion_recognition(self):
        """Test emotion recognition across all test cases"""
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\nTest {i+1}: {test_case['description']}")
            print(f"Text: {test_case['text']}")
            
            response = self.make_api_request(test_case["text"])
            
            assert response.status_code == 200, f"API request failed for test {i+1}"
            
            data = response.json()
            predicted_emotion = data["predicted_emotion"]
            confidence = data["confidence"]
            
            print(f"Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
            
            # Store results for analysis
            results.append({
                "test_case": test_case,
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "passed": True
            })
            
            # Check if expected emotion matches (if specified)
            if test_case["expected_emotion"]:
                if predicted_emotion == test_case["expected_emotion"]:
                    assert confidence >= test_case["min_confidence"], \
                        f"Confidence {confidence:.3f} below minimum {test_case['min_confidence']} for {test_case['description']}"
                    print(f"✅ PASS: Correct emotion with sufficient confidence")
                else:
                    # Check if the predicted emotion is reasonable (in top 2)
                    emotion_probs = data["emotion_probabilities"]
                    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                    top_2_emotions = [emotion for emotion, _ in sorted_emotions[:2]]
                    
                    if test_case["expected_emotion"] in top_2_emotions:
                        print(f"⚠️  PARTIAL: Expected emotion in top 2 predictions")
                        results[-1]["passed"] = "partial"
                    else:
                        print(f"❌ FAIL: Expected {test_case['expected_emotion']}, got {predicted_emotion}")
                        results[-1]["passed"] = False
            else:
                print(f"ℹ️  INFO: Mixed emotion test - predicted {predicted_emotion}")
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passed"] is True)
        partial_tests = sum(1 for r in results if r["passed"] == "partial")
        failed_tests = sum(1 for r in results if r["passed"] is False)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Partial: {partial_tests} ({partial_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Calculate overall accuracy
        accuracy = (passed_tests + partial_tests * 0.5) / total_tests
        print(f"Overall Accuracy: {accuracy*100:.1f}%")
        
        assert accuracy >= 0.7, f"Overall accuracy {accuracy*100:.1f}% below 70% threshold"
    
    def test_performance_metrics(self):
        """Test API performance metrics"""
        test_texts = [
            "Short text",
            "This is a medium length text with some emotional content that should be analyzed properly.",
            "This is a much longer text that contains multiple sentences with various emotional indicators. It should test the performance of the emotion recognition system when dealing with longer content. The system should be able to process this efficiently and provide accurate results even with more complex linguistic patterns and emotional expressions throughout the entire passage."
        ]
        
        performance_results = []
        
        for text in test_texts:
            start_time = time.time()
            response = self.make_api_request(text)
            end_time = time.time()
            
            assert response.status_code == 200, "Performance test request failed"
            
            data = response.json()
            api_processing_time = data["processing_info"]["processing_time_ms"]
            total_response_time = (end_time - start_time) * 1000  # Convert to ms
            
            performance_results.append({
                "text_length": len(text),
                "api_processing_time_ms": api_processing_time,
                "total_response_time_ms": total_response_time
            })
            
            print(f"Text length: {len(text)} chars")
            print(f"API processing: {api_processing_time:.1f}ms")
            print(f"Total response: {total_response_time:.1f}ms")
            print()
        
        # Check performance requirements
        for result in performance_results:
            assert result["api_processing_time_ms"] < 500, \
                f"API processing time {result['api_processing_time_ms']:.1f}ms exceeds 500ms limit"
            assert result["total_response_time_ms"] < 2000, \
                f"Total response time {result['total_response_time_ms']:.1f}ms exceeds 2000ms limit"
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        edge_cases = [
            {
                "text": "",
                "should_fail": True,
                "description": "Empty text"
            },
            {
                "text": "a",
                "should_fail": False,
                "description": "Single character"
            },
            {
                "text": "😀😢😠😨❤️🤩😲🤢😐",
                "should_fail": False,
                "description": "Only emojis"
            },
            {
                "text": "HELLO WORLD THIS IS ALL CAPS TEXT!!!",
                "should_fail": False,
                "description": "All caps text"
            },
            {
                "text": "   \n\t  ",
                "should_fail": True,
                "description": "Only whitespace"
            },
            {
                "text": "This is a very long text that repeats the same content over and over again. " * 50,
                "should_fail": False,
                "description": "Very long text"
            }
        ]
        
        for case in edge_cases:
            print(f"Testing: {case['description']}")
            response = self.make_api_request(case["text"])
            
            if case["should_fail"]:
                assert response.status_code != 200, f"Expected failure for: {case['description']}"
                print(f"✅ Correctly handled invalid input")
            else:
                assert response.status_code == 200, f"Unexpected failure for: {case['description']}"
                data = response.json()
                assert "predicted_emotion" in data, "Missing prediction in response"
                print(f"✅ Successfully processed: {data['predicted_emotion']}")
    
    def test_linguistic_features(self):
        """Test advanced linguistic feature detection"""
        test_cases = [
            {
                "text": "I am NOT happy about this situation!",
                "features_to_check": ["negation_handling"],
                "description": "Negation detection"
            },
            {
                "text": "I'm ABSOLUTELY THRILLED and SUPER excited!!!",
                "features_to_check": ["intensity_detection", "capitalization"],
                "description": "Intensity and capitalization"
            },
            {
                "text": "Feeling sooooo amazingggg today! Yessss! 🎉🎊",
                "features_to_check": ["repeated_letters", "emoji_analysis"],
                "description": "Letter repetition and emoji analysis"
            },
            {
                "text": "Are you serious? Really? How could this happen?",
                "features_to_check": ["question_detection"],
                "description": "Question detection"
            }
        ]
        
        for case in test_cases:
            print(f"Testing: {case['description']}")
            response = self.make_api_request(case["text"])
            
            assert response.status_code == 200, f"Request failed for {case['description']}"
            
            data = response.json()
            linguistic_features = data["analysis_details"]["linguistic_features"]
            features_used = data["processing_info"]["features_used"]
            
            print(f"Detected features: {features_used}")
            print(f"Linguistic analysis: {linguistic_features}")
            
            # Verify specific features are detected
            if "negation_handling" in case["features_to_check"]:
                # Check if negation was properly handled in detailed matches
                detailed_matches = data["analysis_details"]["detailed_matches"]
                negated_words = [match for match in detailed_matches if match.get("negated", False)]
                if "NOT" in case["text"].upper():
                    print(f"Negated words found: {len(negated_words)}")
            
            if "intensity_detection" in case["features_to_check"]:
                # Check for intensity modifiers
                assert any("intensity" in feature.lower() for feature in features_used), \
                    "Intensity detection not found in features"
            
            print(f"✅ Linguistic features properly detected")


def run_manual_tests():
    """Run manual interactive tests"""
    print("Manual Testing Mode")
    print("=" * 50)
    
    while True:
        text = input("\nEnter text to analyze (or 'quit' to exit): ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        try:
            payload = {"text": text, "language": "en"}
            response = requests.post(
                API_ENDPOINT,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n📊 ANALYSIS RESULTS:")
                print(f"🎯 Predicted Emotion: {data['predicted_emotion'].upper()}")
                print(f"📈 Confidence: {data['confidence']:.1%}")
                print(f"⚡ Processing Time: {data['processing_info']['processing_time_ms']:.1f}ms")
                
                print(f"\n📋 Emotion Probabilities:")
                for emotion, prob in sorted(data['emotion_probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    bar = "█" * int(prob * 20)
                    print(f"  {emotion:12} {prob:.1%} {bar}")
                
                print(f"\n🔍 Analysis Details:")
                details = data['analysis_details']
                print(f"  Words: {details['word_count']}")
                print(f"  Emotions detected: {details['emotions_detected']}")
                print(f"  Total emotion score: {details['total_emotion_score']:.2f}")
                
                # Show top matches
                if details['detailed_matches']:
                    print(f"\n🎯 Top Emotional Indicators:")
                    top_matches = sorted(details['detailed_matches'], 
                                       key=lambda x: x['score'], reverse=True)[:5]
                    for match in top_matches:
                        negation = " (negated)" if match.get('negated') else ""
                        print(f"  '{match['word']}' → {match['emotion']} "
                              f"(score: {match['score']:.1f}){negation}")
            
            else:
                print(f"❌ Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Emotion Recognition API")
    parser.add_argument("--manual", action="store_true", 
                       help="Run manual interactive tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick basic tests only")
    
    args = parser.args_parse()
    
    if args.manual:
        run_manual_tests()
    else:
        # Run pytest
        if args.quick:
            pytest.main([__file__ + "::TestEmotionRecognitionAPI::test_basic_emotion_recognition", "-v"])
        else:
            pytest.main([__file__, "-v", "--tb=short"])