"""
Enhanced Text Emotion Recognition API with Improved Accuracy
NeuroFlow - Advanced Emotion Analysis System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import statistics
import json
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroFlow Enhanced Text Emotion Recognition", version="5.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    context: Optional[str] = None

class EmotionResult(BaseModel):
    predicted_emotion: str
    confidence: float
    emotion_probabilities: Dict[str, float]
    analysis_details: Dict
    processing_info: Dict

# Enhanced emotion training data with improved accuracy focus
ENHANCED_EMOTION_DATA = {
    'happy': {
        'primary_keywords': {
            'extreme': ['ecstatic', 'euphoric', 'blissful', 'elated', 'overjoyed', 'exhilarated', 'jubilant', 'exuberant', 'rapturous', 'delighted', 'thrilled'],
            'strong': ['happy', 'joyful', 'cheerful', 'gleeful', 'merry', 'upbeat', 'radiant', 'beaming', 'glowing', 'buoyant', 'spirited'],
            'moderate': ['pleased', 'content', 'satisfied', 'positive', 'good', 'great', 'wonderful', 'nice', 'enjoyable', 'pleasant'],
            'mild': ['okay', 'fine', 'alright', 'decent', 'fair', 'acceptable']
        },
        'context_indicators': ['smile', 'laugh', 'grin', 'chuckle', 'giggle', 'celebrate', 'party', 'success', 'achievement', 'win', 'victory', 'accomplish', 'blessed', 'fortunate', 'lucky'],
        'definitive_phrases': [
            'feel great', 'so happy', 'love it', 'best day', 'amazing time', 'perfect moment', 
            'incredible experience', 'on cloud nine', 'over the moon', 'walking on air', 
            'living the dream', 'couldn\'t be happier', 'absolutely wonderful'
        ],
        'emojis': ['😊', '😄', '😃', '🙂', '😍', '🥰', '😆', '🤗', '🎉', '🎊', '✨', '🌟', '💫', '🌈'],
        'linguistic_patterns': [
            r'\b(so|very|extremely|incredibly|absolutely|really)\s+(happy|joyful|pleased|delighted|thrilled)\b',
            r'\b(love|adore|cherish)\s+(this|it|everything)\b',
            r'\b(best|greatest|most\s+wonderful|amazing|incredible)\s+.*\b',
            r'\b(can\'t|cannot)\s+stop\s+(smiling|laughing|grinning)\b',
            r'\byay+\b|\bhooray\b|\bwoohoo\b'
        ]
    },
    
    'sad': {
        'primary_keywords': {
            'extreme': ['devastated', 'heartbroken', 'despairing', 'anguished', 'crushed', 'shattered', 'grief-stricken', 'inconsolable', 'desolate', 'wretched', 'miserable'],
            'strong': ['depressed', 'sorrowful', 'melancholy', 'dejected', 'downcast', 'forlorn', 'despondent', 'mournful', 'woeful', 'doleful'],
            'moderate': ['sad', 'unhappy', 'blue', 'down', 'low', 'upset', 'disappointed', 'hurt', 'gloomy', 'somber'],
            'mild': ['meh', 'blah', 'dull', 'empty', 'flat', 'subdued', 'listless']
        },
        'context_indicators': ['cry', 'tears', 'weep', 'sob', 'mourn', 'loss', 'grief', 'sorrow', 'pain', 'hurt', 'broken', 'miss', 'longing', 'regret', 'despair', 'funeral', 'death'],
        'definitive_phrases': [
            'feel down', 'so sad', 'broken heart', 'terrible day', 'worst time', 
            'painful moment', 'feel empty', 'lost everything', 'can\'t go on',
            'want to cry', 'makes me cry', 'heartbreaking', 'devastating news'
        ],
        'emojis': ['😢', '😭', '😞', '☹️', '🙁', '😔', '😟', '😦', '💔', '😿', '😪', '😫'],
        'linguistic_patterns': [
            r'\b(so|very|extremely|incredibly|really)\s+(sad|depressed|upset|hurt|heartbroken)\b',
            r'\b(can\'t|cannot)\s+stop\s+(crying|weeping|sobbing)\b',
            r'\b(feel|feeling)\s+(empty|hollow|lost|broken|devastated)\b',
            r'\b(worst|terrible|horrible|awful)\s+.*\b',
            r'\bheart\s*break\w*\b|\bbroken\s+heart\b'
        ]
    },
    
    'angry': {
        'primary_keywords': {
            'extreme': ['furious', 'enraged', 'livid', 'incensed', 'outraged', 'irate', 'seething', 'infuriated', 'blazing', 'wrathful'],
            'strong': ['angry', 'mad', 'pissed', 'irritated', 'frustrated', 'aggravated', 'vexed', 'steaming', 'indignant', 'resentful'],
            'moderate': ['annoyed', 'bothered', 'displeased', 'cross', 'grumpy', 'cranky', 'ticked', 'miffed', 'irked', 'perturbed'],
            'mild': ['slightly annoyed', 'somewhat bothered', 'a bit irritated', 'mildly upset']
        },
        'context_indicators': ['hate', 'rage', 'fury', 'wrath', 'fight', 'argue', 'yell', 'shout', 'scream', 'slam', 'violence', 'hostile', 'aggressive', 'attack'],
        'definitive_phrases': [
            'so angry', 'fed up', 'had enough', 'losing it', 'boiling mad', 
            'seeing red', 'ready to explode', 'last straw', 'driving me crazy',
            'makes me angry', 'pissed off', 'sick of this', 'can\'t stand'
        ],
        'emojis': ['😠', '😡', '🤬', '👿', '💢', '😤', '🔥', '⚡', '💥', '🗯️'],
        'linguistic_patterns': [
            r'\b(so|very|extremely|incredibly|really)\s+(angry|mad|furious|pissed|frustrated)\b',
            r'\b(can\'t|cannot)\s+stand\s+(this|it|them)\b',
            r'\b(fed|sick)\s+up\s+(with|of)\b',
            r'\b(makes?\s+me|making\s+me)\s+(angry|mad|furious)\b',
            r'\b(what\s+the\s+(hell|fuck|damn))\b|\bdamn\s+it\b'
        ]
    },
    
    'fear': {
        'primary_keywords': {
            'extreme': ['terrified', 'petrified', 'horrified', 'panic-stricken', 'terror-stricken', 'paralyzed', 'traumatized'],
            'strong': ['scared', 'afraid', 'frightened', 'fearful', 'alarmed', 'startled', 'spooked', 'panicked', 'distressed'],
            'moderate': ['nervous', 'anxious', 'worried', 'apprehensive', 'concerned', 'uneasy', 'tense', 'cautious', 'wary'],
            'mild': ['uncertain', 'doubtful', 'uncomfortable', 'unsure', 'hesitant', 'reluctant']
        },
        'context_indicators': ['panic', 'dread', 'phobia', 'nightmare', 'terror', 'fright', 'anxiety', 'worry', 'threat', 'danger', 'risk', 'scared', 'afraid'],
        'definitive_phrases': [
            'so scared', 'really afraid', 'panic attack', 'worst fear', 'terrifying moment', 
            'can\'t handle', 'too scared', 'frightening experience', 'scared to death',
            'shaking with fear', 'heart racing', 'nervous breakdown'
        ],
        'emojis': ['😨', '😰', '😱', '🙀', '😧', '😖', '💀', '👻', '🆘', '⚠️'],
        'linguistic_patterns': [
            r'\b(so|very|extremely|incredibly|really)\s+(scared|afraid|terrified|worried|nervous)\b',
            r'\b(can\'t|cannot)\s+(handle|take|deal\s+with)\s+(this|it)\b',
            r'\b(worst|deepest|greatest)\s+fear\b',
            r'\b(panic|anxiety)\s+attack\b',
            r'\bscared\s+(to\s+death|out\s+of\s+my\s+mind)\b'
        ]
    },
    
    'love': {
        'primary_keywords': {
            'extreme': ['adore', 'worship', 'idolize', 'cherish', 'treasure', 'devoted', 'passionate', 'unconditional'],
            'strong': ['love', 'romantic', 'intimate', 'affectionate', 'beloved', 'darling', 'sweetheart', 'precious'],
            'moderate': ['like', 'fond', 'care', 'appreciate', 'value', 'enjoy', 'special', 'important', 'meaningful'],
            'mild': ['nice', 'sweet', 'cute', 'lovely', 'pleasant', 'agreeable']
        },
        'context_indicators': ['heart', 'romance', 'kiss', 'hug', 'embrace', 'valentine', 'marriage', 'wedding', 'relationship', 'partner', 'couple', 'soulmate'],
        'definitive_phrases': [
            'love you', 'so in love', 'perfect match', 'soulmate', 'meant to be', 
            'head over heels', 'crazy about', 'deeply in love', 'love of my life',
            'can\'t live without', 'complete me', 'better half'
        ],
        'emojis': ['❤️', '💕', '💖', '💗', '💘', '💝', '😍', '🥰', '💋', '💏', '💑'],
        'linguistic_patterns': [
            r'\b(love|adore|cherish)\s+(you|him|her|them)\b',
            r'\b(so|very|deeply|madly)\s+in\s+love\b',
            r'\b(perfect|ideal|true)\s+(match|love|partner)\b',
            r'\b(head\s+over\s+heels|crazy\s+about|smitten\s+with)\b'
        ]
    },
    
    'excitement': {
        'primary_keywords': {
            'extreme': ['exhilarated', 'electrified', 'energized', 'pumped', 'fired up', 'hyped', 'buzzing', 'charged', 'amped'],
            'strong': ['excited', 'thrilled', 'enthusiastic', 'eager', 'animated', 'spirited', 'vibrant', 'dynamic'],
            'moderate': ['interested', 'keen', 'ready', 'motivated', 'inspired', 'anticipating'],
            'mild': ['curious', 'intrigued', 'wondering', 'hopeful', 'optimistic']
        },
        'context_indicators': ['adventure', 'party', 'event', 'celebration', 'festival', 'concert', 'game', 'match', 'opportunity', 'chance'],
        'definitive_phrases': [
            'so excited', 'can\'t wait', 'looking forward', 'pumped up', 'ready to go', 
            'bring it on', 'let\'s do this', 'super excited', 'really excited',
            'bursting with excitement', 'jumping up and down'
        ],
        'emojis': ['🤩', '🔥', '⚡', '🎉', '🎊', '🙌', '💪', '🚀', '✨', '⭐'],
        'linguistic_patterns': [
            r'\b(so|very|extremely|incredibly|really|super)\s+(excited|thrilled|pumped|hyped)\b',
            r'\b(can\'t|cannot)\s+wait\s+(for|to)\b',
            r'\b(looking\s+forward|excited\s+about)\b',
            r'\b(bring\s+it\s+on|let\'s\s+do\s+this|ready\s+to\s+go)\b'
        ]
    },
    
    'surprise': {
        'primary_keywords': {
            'extreme': ['astonished', 'astounded', 'flabbergasted', 'stunned', 'dumbfounded', 'thunderstruck', 'blown away', 'mind-blown'],
            'strong': ['surprised', 'shocked', 'amazed', 'startled', 'bewildered', 'confounded', 'speechless'],
            'moderate': ['unexpected', 'sudden', 'abrupt', 'unforeseen', 'curious', 'interesting', 'remarkable'],
            'mild': ['huh', 'oh', 'really', 'wow', 'hmm', 'interesting']
        },
        'context_indicators': ['sudden', 'unexpected', 'out of nowhere', 'all of a sudden', 'without warning', 'plot twist', 'revelation'],
        'definitive_phrases': [
            'can\'t believe', 'no way', 'what a surprise', 'didn\'t see that coming', 
            'totally unexpected', 'out of the blue', 'caught off guard',
            'never expected', 'what a shock', 'completely surprised'
        ],
        'emojis': ['😲', '😮', '🤯', '😯', '🙊', '‼️', '❗', '⁉️', '🤔'],
        'linguistic_patterns': [
            r'\b(can\'t|cannot)\s+believe\b',
            r'\b(no\s+way|what\s+a\s+surprise)\b',
            r'\b(didn\'t\s+see|never\s+expected)\b',
            r'\b(out\s+of\s+(nowhere|the\s+blue))\b',
            r'\b(what|how)\s+(a\s+)?(shock|surprise)\b'
        ]
    },
    
    'disgust': {
        'primary_keywords': {
            'extreme': ['revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'abhorred', 'detested', 'loathed'],
            'strong': ['disgusted', 'disgusting', 'gross', 'revolting', 'repugnant', 'loathsome', 'vile', 'foul'],
            'moderate': ['awful', 'terrible', 'horrible', 'nasty', 'yucky', 'unpleasant', 'offensive'],
            'mild': ['ew', 'yuck', 'ick', 'bleh', 'weird', 'strange']
        },
        'context_indicators': ['vomit', 'puke', 'sick', 'nausea', 'rotten', 'stink', 'smell', 'filthy', 'dirty', 'contaminated'],
        'definitive_phrases': [
            'makes me sick', 'so gross', 'absolutely disgusting', 'can\'t stand it', 
            'turns my stomach', 'want to throw up', 'makes me nauseous',
            'completely revolting', 'utterly disgusting'
        ],
        'emojis': ['🤢', '🤮', '😷', '🤧', '💩', '🗑️', '👎', '🙅'],
        'linguistic_patterns': [
            r'\b(makes?\s+me|making\s+me)\s+(sick|nauseous|want\s+to\s+throw\s+up)\b',
            r'\b(so|very|extremely|absolutely|completely)\s+(gross|disgusting|revolting)\b',
            r'\b(can\'t|cannot)\s+stand\s+(it|this|that)\b'
        ]
    },
    
    'neutral': {
        'primary_keywords': {
            'extreme': [],
            'strong': ['normal', 'regular', 'standard', 'typical', 'ordinary', 'average', 'usual', 'routine'],
            'moderate': ['okay', 'fine', 'alright', 'so-so', 'decent', 'fair', 'moderate', 'balanced'],
            'mild': ['whatever', 'meh', 'eh', 'sure', 'maybe', 'perhaps', 'possibly']
        },
        'context_indicators': ['normal', 'usual', 'typical', 'routine', 'standard', 'regular', 'ordinary', 'common'],
        'definitive_phrases': [
            'it\'s okay', 'not bad', 'could be worse', 'nothing special', 
            'just normal', 'same as usual', 'pretty average', 'nothing new'
        ],
        'emojis': ['😐', '😑', '🙂', '😶', '🤷', '🤔'],
        'linguistic_patterns': [
            r'\b(just|pretty|fairly)\s+(normal|ordinary|typical|average)\b',
            r'\b(nothing\s+special|same\s+as\s+usual)\b',
            r'\b(it\'s\s+)?(okay|fine|alright)\b'
        ]
    }
}

class EnhancedTextAnalyzer:
    """Enhanced text emotion analyzer with improved accuracy"""
    
    def __init__(self):
        self.emotions = list(ENHANCED_EMOTION_DATA.keys())
        
        # Emotion opposites for negation handling
        self.emotion_opposites = {
            'happy': 'sad',
            'sad': 'happy', 
            'angry': 'love',
            'love': 'angry',
            'fear': 'excitement',
            'excitement': 'fear',
            'surprise': 'neutral',
            'disgust': 'love',
            'neutral': 'neutral'
        }
        
        # Intensity modifiers with stronger weights
        self.intensity_modifiers = {
            'extreme': ['extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly', 'supremely'],
            'high': ['very', 'really', 'quite', 'pretty', 'rather', 'truly', 'deeply', 'highly'],
            'moderate': ['somewhat', 'fairly', 'reasonably', 'kind of', 'sort of', 'a bit'],
            'low': ['slightly', 'barely', 'hardly', 'scarcely', 'a little']
        }
        
        # Negation patterns with better coverage
        self.negation_patterns = [
            r'\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b',
            r'\b(don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|cannot|couldn\'t|shouldn\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\b',
            r'\b(hardly|barely|scarcely|rarely|seldom)\b'
        ]
    
    def preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing"""
        # Handle contractions
        contractions = {
            "can't": "cannot", "won't": "will not", "n't": " not",
            "i'm": "i am", "you're": "you are", "we're": "we are",
            "they're": "they are", "it's": "it is", "that's": "that is"
        }
        
        processed_text = text.lower()
        for contraction, expansion in contractions.items():
            processed_text = processed_text.replace(contraction, expansion)
        
        # Split into words and clean
        words = re.findall(r'\b\w+\b', processed_text)
        return words
    
    def detect_negation(self, words: List[str], position: int, window: int = 3) -> bool:
        """Enhanced negation detection with larger context window"""
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        context = ' '.join(words[start:end])
        
        for pattern in self.negation_patterns:
            if re.search(pattern, context):
                return True
        return False
    
    def detect_intensity(self, words: List[str], position: int, window: int = 2) -> float:
        """Enhanced intensity detection"""
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        context_words = words[start:end]
        
        max_intensity = 1.0
        
        for word in context_words:
            for level, modifiers in self.intensity_modifiers.items():
                if word in modifiers:
                    if level == 'extreme':
                        max_intensity = max(max_intensity, 2.5)
                    elif level == 'high':
                        max_intensity = max(max_intensity, 2.0)
                    elif level == 'moderate':
                        max_intensity = max(max_intensity, 1.5)
                    elif level == 'low':
                        max_intensity = max(max_intensity, 0.8)
        
        # Check for repeated punctuation (!!!, ???)
        if position < len(words) - 1:
            next_chars = ' '.join(words[position:position+2])
            if re.search(r'[!]{2,}', next_chars):
                max_intensity *= 1.3
            elif re.search(r'[?]{2,}', next_chars):
                max_intensity *= 1.2
        
        return max_intensity
    
    def extract_linguistic_features(self, text: str, words: List[str]) -> Dict:
        """Extract enhanced linguistic features"""
        return {
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'avg_word_length': statistics.mean([len(word) for word in words if word.isalpha()]) if words else 0,
            'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text)),
            'repeated_letters': len(re.findall(r'(\w)\1{2,}', text)),
            'all_caps_words': len([word for word in words if word.isupper() and len(word) > 1]),
            'word_count': len([w for w in words if w.isalpha()]),
            'punctuation_density': len(re.findall(r'[.!?;,:]', text)) / max(len(words), 1)
        }
    
    def analyze_emotion_patterns(self, text: str, words: List[str]) -> Tuple[Dict[str, float], List[Dict]]:
        """Enhanced emotion pattern analysis with better accuracy"""
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        detailed_matches = []
        text_lower = text.lower()
        
        for emotion, data in ENHANCED_EMOTION_DATA.items():
            emotion_score = 0.0
            
            # 1. Primary keyword analysis with enhanced scoring
            for i, word in enumerate(words):
                if not word.isalpha() or len(word) < 2:
                    continue
                    
                word_lower = word.lower()
                word_score = 0
                match_type = None
                
                # Check primary keywords with precise scoring
                for level, keywords in data['primary_keywords'].items():
                    if word_lower in keywords:
                        if level == 'extreme':
                            word_score = 10  # Highest weight for extreme words
                        elif level == 'strong':
                            word_score = 7
                        elif level == 'moderate':
                            word_score = 4
                        elif level == 'mild':
                            word_score = 2
                        match_type = f"{level}_keyword"
                        break
                
                # Check context indicators
                if word_score == 0:
                    for indicator in data['context_indicators']:
                        if indicator in word_lower or word_lower in indicator:
                            word_score = 3
                            match_type = "context_indicator"
                            break
                
                if word_score > 0:
                    # Apply intensity modifiers
                    intensity = self.detect_intensity(words, i)
                    original_score = word_score
                    word_score *= intensity
                    
                    # Handle negation with sophisticated logic
                    is_negated = self.detect_negation(words, i)
                    if is_negated:
                        # Flip to opposite emotion for negated strong emotions
                        if original_score >= 7:  # Strong emotions
                            opposite = self.emotion_opposites.get(emotion, 'neutral')
                            if opposite != 'neutral':
                                emotion_scores[opposite] += word_score * 0.7
                            word_score *= -0.3  # Negative contribution to current emotion
                        else:
                            word_score *= 0.3  # Just reduce for weaker emotions
                    
                    emotion_score += word_score
                    
                    detailed_matches.append({
                        'word': word_lower,
                        'emotion': emotion,
                        'score': abs(word_score),
                        'type': match_type,
                        'intensity': intensity,
                        'negated': is_negated,
                        'position': i
                    })
            
            # 2. Definitive phrase matching (highest reliability)
            for phrase in data['definitive_phrases']:
                if phrase.lower() in text_lower:
                    phrase_score = 8  # High score for definitive phrases
                    emotion_score += phrase_score
                    detailed_matches.append({
                        'word': phrase,
                        'emotion': emotion,
                        'score': phrase_score,
                        'type': 'definitive_phrase',
                        'intensity': 1.0,
                        'negated': False,
                        'position': -1
                    })
            
            # 3. Emoji analysis (very reliable indicators)
            for emoji in data['emojis']:
                emoji_count = text.count(emoji)
                if emoji_count > 0:
                    emoji_score = emoji_count * 6  # High weight for emojis
                    emotion_score += emoji_score
                    detailed_matches.append({
                        'word': emoji,
                        'emotion': emotion,
                        'score': emoji_score,
                        'type': 'emoji_match',
                        'intensity': 1.0,
                        'negated': False,
                        'position': -1
                    })
            
            # 4. Linguistic pattern matching (very reliable)
            for pattern in data['linguistic_patterns']:
                try:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        pattern_score = len(matches) * 9  # Very high weight for patterns
                        emotion_score += pattern_score
                        detailed_matches.append({
                            'word': f"pattern_match",
                            'emotion': emotion,
                            'score': pattern_score,
                            'type': 'linguistic_pattern',
                            'intensity': 1.0,
                            'negated': False,
                            'position': -1
                        })
                except re.error:
                    continue
            
            # 5. Multi-evidence boost
            evidence_types = set(match['type'] for match in detailed_matches if match['emotion'] == emotion)
            if len(evidence_types) >= 2:
                multi_evidence_boost = emotion_score * 0.25  # 25% boost for multiple evidence
                emotion_score += multi_evidence_boost
            
            emotion_scores[emotion] = max(0, emotion_score)
        
        return emotion_scores, detailed_matches
    
    def calculate_enhanced_confidence(self, emotion_scores: Dict, detailed_matches: List, 
                                    linguistic_features: Dict, word_count: int) -> float:
        """Enhanced confidence calculation with better accuracy indicators"""
        max_score = max(emotion_scores.values()) if emotion_scores.values() else 0
        if max_score == 0:
            return 0.5
        
        # Base confidence from score strength and word density
        score_density = max_score / max(word_count, 1)
        base_confidence = min(0.85, 0.3 + score_density * 0.2)
        
        # High-reliability evidence boosts
        confidence_boosts = []
        
        # Definitive phrases are most reliable
        definitive_matches = sum(1 for match in detailed_matches if match['type'] == 'definitive_phrase')
        if definitive_matches > 0:
            confidence_boosts.append(min(0.20, definitive_matches * 0.15))
        
        # Linguistic patterns are very reliable
        pattern_matches = sum(1 for match in detailed_matches if match['type'] == 'linguistic_pattern')
        if pattern_matches > 0:
            confidence_boosts.append(0.15)
        
        # Emoji matches are reliable
        emoji_matches = sum(1 for match in detailed_matches if match['type'] == 'emoji_match')
        if emoji_matches > 0:
            confidence_boosts.append(min(0.12, emoji_matches * 0.06))
        
        # Multiple evidence types increase confidence
        evidence_types = set(match['type'] for match in detailed_matches)
        if len(evidence_types) >= 3:
            confidence_boosts.append(0.10)
        elif len(evidence_types) >= 2:
            confidence_boosts.append(0.05)
        
        # High-intensity indicators
        high_intensity = sum(1 for match in detailed_matches if match['intensity'] >= 2.0)
        if high_intensity > 0:
            confidence_boosts.append(min(0.08, high_intensity * 0.03))
        
        # Strong emotional punctuation
        if linguistic_features['exclamation_count'] >= 2:
            confidence_boosts.append(0.06)
        
        # Apply boosts
        boosted_confidence = base_confidence + sum(confidence_boosts)
        
        # Confidence penalties
        penalties = []
        
        # Mixed emotions penalty
        sorted_scores = sorted(emotion_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > sorted_scores[0] * 0.4:
            penalties.append(0.08)
        
        # Very short text penalty
        if word_count < 3:
            penalties.append(0.12)
        
        # Too many negations might indicate confusion
        negated_matches = sum(1 for match in detailed_matches if match.get('negated', False))
        if negated_matches > len(detailed_matches) * 0.3:
            penalties.append(0.05)
        
        final_confidence = boosted_confidence - sum(penalties)
        return max(0.25, min(0.95, final_confidence))
    
    def analyze_text(self, text: str, context: str = None) -> Dict:
        """Complete enhanced text emotion analysis"""
        if not text or not text.strip():
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.5,
                'emotion_probabilities': {emotion: 1/len(self.emotions) for emotion in self.emotions},
                'analysis_details': {
                    'word_count': 0,
                    'total_emotion_score': 0,
                    'dominant_score': 0,
                    'linguistic_features': {},
                    'detailed_matches': [],
                    'emotions_detected': 0
                },
                'processing_info': {
                    'processing_time_ms': 0,
                    'algorithm_version': '5.0',
                    'features_used': []
                }
            }
        
        start_time = datetime.now()
        
        # Preprocess and analyze
        words = self.preprocess_text(text)
        word_count = len([w for w in words if w.isalpha()])
        linguistic_features = self.extract_linguistic_features(text, words)
        emotion_scores, detailed_matches = self.analyze_emotion_patterns(text, words)
        
        # Calculate confidence
        confidence = self.calculate_enhanced_confidence(
            emotion_scores, detailed_matches, linguistic_features, word_count
        )
        
        # Determine predicted emotion
        max_score = max(emotion_scores.values()) if emotion_scores.values() else 0
        if max_score <= 0:
            predicted_emotion = 'neutral'
            confidence = 0.5
        else:
            predicted_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Calculate normalized probabilities
        total_score = sum(emotion_scores.values())
        if total_score == 0:
            emotion_probabilities = {emotion: 1/len(self.emotions) for emotion in self.emotions}
        else:
            # Apply softmax-like normalization for better probability distribution
            emotion_probabilities = {}
            for emotion, score in emotion_scores.items():
                normalized_score = max(0.01, score / total_score)
                emotion_probabilities[emotion] = normalized_score
            
            # Renormalize to ensure sum = 1
            total_prob = sum(emotion_probabilities.values())
            emotion_probabilities = {k: v/total_prob for k, v in emotion_probabilities.items()}
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'emotion_probabilities': emotion_probabilities,
            'analysis_details': {
                'word_count': word_count,
                'total_emotion_score': total_score,
                'dominant_score': max_score,
                'linguistic_features': linguistic_features,
                'detailed_matches': detailed_matches,
                'emotions_detected': len([score for score in emotion_scores.values() if score > 0])
            },
            'processing_info': {
                'processing_time_ms': round(processing_time, 2),
                'algorithm_version': '5.0',
                'features_used': [
                    'enhanced_keywords', 'context_indicators', 'definitive_phrases',
                    'linguistic_patterns', 'emoji_analysis', 'intensity_modifiers',
                    'negation_detection', 'multi_evidence_boost', 'confidence_scoring'
                ]
            }
        }

# Global analyzer instance
enhanced_analyzer = EnhancedTextAnalyzer()

@app.post("/api/emotion/analyze-text", response_model=EmotionResult)
async def analyze_text_emotion(request: TextRequest):
    """Enhanced text emotion analysis with improved accuracy"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = enhanced_analyzer.analyze_text(request.text, request.context)
        
        logger.info(f"Enhanced analysis: '{request.text[:50]}...' -> {result['predicted_emotion']} ({result['confidence']:.2f})")
        
        return EmotionResult(**result)
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "5.0.0"}

@app.get("/")
async def root():
    return {
        "message": "NeuroFlow Enhanced Text Emotion Recognition API v5.0",
        "description": "Advanced text emotion analysis with improved accuracy and reliability",
        "features": [
            "9 Comprehensive Emotion Categories",
            "Enhanced Linguistic Pattern Recognition",
            "Advanced Negation Handling",
            "Multi-Evidence Confidence Scoring",
            "Definitive Phrase Matching",
            "Context-Aware Analysis",
            "Emoji & Symbol Analysis",
            "High Accuracy Processing"
        ],
        "supported_emotions": list(ENHANCED_EMOTION_DATA.keys()),
        "accuracy_improvements": [
            "Better keyword weighting system",
            "Enhanced negation detection",
            "Multi-evidence validation",
            "Definitive phrase patterns",
            "Improved confidence calculation",
            "Context-aware processing"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)