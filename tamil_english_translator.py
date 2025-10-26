#!/usr/bin/env python3
"""
Tamil to English Translator
Translates Tamil text to English without using external APIs
Built-in translation dictionary and NLP techniques
"""

import json
import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict

class TamilEnglishTranslator:
    """
    Translates Tamil text to English using built-in dictionaries and NLP
    """
    
    def __init__(self):
        """Initialize with Tamil-English word mappings"""

        # Comprehensive Tamil to English dictionary with word categories
        self.tamil_english_dict = {
            # Common nouns
            'கால': 'time/leg',
            'கல்': 'stone',
            'கர்': 'hand',
            'கவி': 'poet',
            'கை': 'hand',
            'கோ': 'cow',
            'சக': 'friend',
            'சல': 'water',
            'சாக': 'proof',
            'சரி': 'correct/right',
            'சிறு': 'small',
            'சுக': 'happiness',
            'சுவை': 'taste',
            'தக': 'suitable',
            'தல': 'place',
            'தாய்': 'mother',
            'தி': 'day',
            'தீ': 'fire',
            'தெய்': 'god',
            'தொல்': 'ancient',
            'நக': 'nail',
            'நல': 'good',
            'நாய்': 'dog',
            'நீ': 'you',
            'நீர்': 'water',
            'பக': 'side',
            'பல': 'many',
            'பாய்': 'run',
            'பாவ': 'sin',
            'பி': 'after',
            'பொல': 'lie',
            'மக': 'son',
            'மல': 'flower',
            'மாய': 'illusion',
            'முக': 'face',
            'முல': 'first',
            'யக': 'yoke',
            'ரக': 'taste',
            'ரல': 'sound',
            'ரா': 'night',
            'ரி': 'debt',
            'வக': 'class',
            'வல': 'strength',
            'வாய்': 'mouth',
            'வி': 'spread',
            'வீ': 'house',
            'வெய்': 'sun',
            'வொல': 'word',

            # Verbs
            'சாய': 'lean',
            'பாய்': 'run',
            'வி': 'spread',
            'செய்': 'do',
            'கொ': 'take',
            'வா': 'come',
            'போ': 'go',
            'இரு': 'be',
            'உண்': 'eat',
            'குடி': 'drink',
            'பேசு': 'speak',
            'கேட்': 'listen',
            'பார்': 'see',
            'எழு': 'write',
            'படி': 'read',
            'நட': 'walk',
            'ஓட': 'run',
            'விழு': 'fall',
            'எழு': 'rise',
            'தூங்கு': 'sleep',
            'விழி': 'wake',

            # Adjectives
            'பெரிய': 'big',
            'சிறிய': 'small',
            'நல்ல': 'good',
            'கெட்ட': 'bad',
            'வெள்ளை': 'white',
            'கருப்பு': 'black',
            'சிவப்பு': 'red',
            'பச்சை': 'green',
            'மஞ்சள்': 'yellow',
            'நீலம்': 'blue',
            'புதிய': 'new',
            'பழைய': 'old',
            'வேகமான': 'fast',
            'மெதுவான': 'slow',
            'சூடான': 'hot',
            'குளிர்ந்த': 'cold',
            'ஈரமான': 'wet',
            'உலர்ந்த': 'dry',
            'மென்மையான': 'soft',
            'கடினமான': 'hard',

            # Pronouns
            'நான்': 'I',
            'நீ': 'you',
            'அவன்': 'he',
            'அவள்': 'she',
            'அது': 'it',
            'நாம்': 'we',
            'நீங்கள்': 'you(plural)',
            'அவர்': 'they',

            # Prepositions
            'மேல்': 'on',
            'கீழ்': 'under',
            'உள்': 'inside',
            'வெளி': 'outside',
            'முன்': 'before',
            'பின்': 'after',
            'பக்கம்': 'beside',
            'இடையே': 'between',
            'மூலம்': 'through',
            'வழியே': 'via',

            # Common words
            'ஆம்': 'yes',
            'இல்லை': 'no',
            'என்': 'what',
            'யார்': 'who',
            'எங்கே': 'where',
            'எப்போது': 'when',
            'ஏன்': 'why',
            'எப்படி': 'how',
            'மற்றும்': 'and',
            'அல்லது': 'or',
            'ஆனால்': 'but',
            'ஏனெனில்': 'because',
            'அதனால்': 'therefore',
            'இதனால்': 'thus',
            'ஆகவே': 'so',
            'மிகவும்': 'very',
            'கொஞ்சம்': 'little',
            'நிறைய': 'lot',
            'எல்லாம்': 'all',
            'சில': 'some',
            'ஒன்று': 'one',
            'இரண்டு': 'two',
            'மூன்று': 'three',
            'நான்கு': 'four',
            'ஐந்து': 'five',
        }

        # Enhanced dictionary with word categories and meanings for literal translation
        self.tamil_word_meanings = {
            # Greetings and common phrases
            'வணக்கம்': {
                'english': 'hello',
                'category': 'noun',
                'meaning': 'greeting/salutation',
                'explanation': 'Tamil equivalent of "Hi" or "Hello"',
                'usage': 'formal'
            },
            'நல்ல': {
                'english': 'good',
                'category': 'adjective',
                'meaning': 'positive quality',
                'explanation': 'Describes something positive or well',
                'usage': 'common'
            },
            'நன்றி': {
                'english': 'thank you',
                'category': 'noun',
                'meaning': 'gratitude/thanks',
                'explanation': 'Tamil equivalent of "Thank you"',
                'usage': 'formal'
            },
            'சரி': {
                'english': 'okay/correct',
                'category': 'adjective',
                'meaning': 'right/correct/fine',
                'explanation': 'Means "okay" or "correct"',
                'usage': 'common'
            },
            'நீங்க': {
                'english': 'you',
                'category': 'pronoun',
                'meaning': 'formal you',
                'explanation': 'Formal way to address someone',
                'usage': 'formal'
            },
            'நீங்கள்': {
                'english': 'you',
                'category': 'pronoun',
                'meaning': 'formal you (plural)',
                'explanation': 'Formal way to address multiple people',
                'usage': 'formal'
            },
            'எப்படி': {
                'english': 'how',
                'category': 'adverb',
                'meaning': 'in what manner',
                'explanation': 'Question word asking about manner or condition',
                'usage': 'common'
            },
            'இருக்கீங்க': {
                'english': 'are you',
                'category': 'verb',
                'meaning': 'present tense of "to be"',
                'explanation': 'Asking about someone\'s state or condition',
                'usage': 'formal'
            },
            'இருக்கிறீர்கள்': {
                'english': 'are you',
                'category': 'verb',
                'meaning': 'present tense of "to be" (formal)',
                'explanation': 'Formal way of asking "how are you"',
                'usage': 'formal'
            },
            'நான்': {
                'english': 'I',
                'category': 'pronoun',
                'meaning': 'first person singular',
                'explanation': 'Refers to oneself',
                'usage': 'common'
            },
            'என்': {
                'english': 'my',
                'category': 'pronoun',
                'meaning': 'possessive first person',
                'explanation': 'Shows possession by the speaker',
                'usage': 'common'
            },
            'பெயர்': {
                'english': 'name',
                'category': 'noun',
                'meaning': 'what someone is called',
                'explanation': 'Refers to a person\'s name',
                'usage': 'common'
            },
            'என்ன': {
                'english': 'what',
                'category': 'pronoun',
                'meaning': 'question word',
                'explanation': 'Asking for information',
                'usage': 'common'
            },
            'செய்கிறீர்கள்': {
                'english': 'are doing',
                'category': 'verb',
                'meaning': 'present continuous of "to do"',
                'explanation': 'Asking what someone is currently doing',
                'usage': 'formal'
            },
            'செய்கிறீங்க': {
                'english': 'are doing',
                'category': 'verb',
                'meaning': 'present continuous of "to do"',
                'explanation': 'Asking what someone is currently doing',
                'usage': 'formal'
            },
            'வேலை': {
                'english': 'work/job',
                'category': 'noun',
                'meaning': 'occupation or task',
                'explanation': 'Refers to work or employment',
                'usage': 'common'
            },
            'வீடு': {
                'english': 'house/home',
                'category': 'noun',
                'meaning': 'place of residence',
                'explanation': 'Refers to a home or house',
                'usage': 'common'
            },
            'குடும்பம்': {
                'english': 'family',
                'category': 'noun',
                'meaning': 'group of relatives',
                'explanation': 'Refers to family members',
                'usage': 'common'
            },
            'நல்ல': {
                'english': 'good',
                'category': 'adjective',
                'meaning': 'positive quality',
                'explanation': 'Describes something positive',
                'usage': 'common'
            },
            'கெட்ட': {
                'english': 'bad',
                'category': 'adjective',
                'meaning': 'negative quality',
                'explanation': 'Describes something negative',
                'usage': 'common'
            },
            'பெரிய': {
                'english': 'big',
                'category': 'adjective',
                'meaning': 'large in size',
                'explanation': 'Describes something large',
                'usage': 'common'
            },
            'சிறிய': {
                'english': 'small',
                'category': 'adjective',
                'meaning': 'small in size',
                'explanation': 'Describes something small',
                'usage': 'common'
            },
        }
        
        # Character-level mappings for partial matches
        self.char_patterns = {
            'ர': 'r',
            'ல': 'l',
            'ன': 'n',
            'ம': 'm',
            'ய': 'y',
            'வ': 'v',
            'ப': 'p',
            'த': 't',
            'க': 'k',
            'ச': 'ch',
            'ட': 'd',
            'ஞ': 'ny',
            'ங': 'ng',
            'ஷ': 'sh',
            'ஸ': 's',
            'ஹ': 'h',
            'ள': 'll',
            'ற': 'rr',
            'ந': 'nh',
            'ண': 'nn',
        }
        
        print("✓ Tamil-English Translator initialized")
        print(f"✓ Loaded {len(self.tamil_word_meanings)} word meanings for literal translation")

        # Comprehensive Tamil to Roman transliteration mapping
        # Using a phonetic-based scheme for accurate transliteration
        self.tamil_to_roman = {
            # Consonants (with inherent 'a' sound)
            'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஞ': 'nja',
            'ட': 'ta', 'ண': 'na', 'த': 'tha', 'ந': 'na',
            'ப': 'pa', 'ம': 'ma', 'ய': 'ya', 'ர': 'ra',
            'ல': 'la', 'வ': 'va', 'ழ': 'zha', 'ள': 'la',
            'ற': 'ra', 'ன': 'na',

            # Standalone vowels
            'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii',
            'உ': 'u', 'ஊ': 'uu', 'எ': 'e', 'ஏ': 'ee',
            'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',

            # Vowel signs (matras) - modify the preceding consonant
            'ா': 'aa', 'ி': 'i', 'ீ': 'ii', 'ு': 'u',
            'ூ': 'uu', 'ெ': 'e', 'ே': 'ee', 'ை': 'ai',
            'ொ': 'o', 'ோ': 'oo', 'ௌ': 'au',

            # Special characters
            '்': '',  # Virama (no vowel sound)
            'ஃ': 'h',  # Anusvara
        }

        # Character-level meanings (for individual Tamil characters)
        self.character_meanings = {
            'ர': 'r/sound',
            'ந': 'n/nose',
            'ு': 'u/vowel',
            'ல': 'l/tongue',
            'க': 'k/throat',
            'ச': 'ch/palate',
            'த': 'th/teeth',
            'ப': 'p/lips',
            'ம': 'm/mouth',
            'ய': 'y/semi-vowel',
            'வ': 'v/lips',
            'ள': 'l/retroflex',
            'ற': 'r/retroflex',
            'ன': 'n/retroflex',
            'ட': 't/retroflex',
            'ு': 'u/vowel',
            'ி': 'i/vowel',
            'ா': 'aa/vowel',
            'ே': 'e/vowel',
            'ை': 'ai/vowel',
            'ொ': 'o/vowel',
            'ோ': 'o/vowel',
            'ௌ': 'au/vowel',
        }

        print(f"✓ Loaded {len(self.tamil_english_dict)} word mappings")
        print(f"✓ Loaded {len(self.tamil_to_roman)} transliteration mappings")
        print(f"✓ Loaded {len(self.character_meanings)} character meanings")
    
    def translate_word(self, tamil_word: str) -> str:
        """
        Translate a single Tamil word to English
        
        Args:
            tamil_word: Tamil word to translate
            
        Returns:
            English translation or transliteration
        """
        tamil_word = tamil_word.strip()

        # Direct lookup
        if tamil_word in self.tamil_english_dict:
            return self.tamil_english_dict[tamil_word]

        # Check if it's a single character with meaning
        if tamil_word in self.character_meanings:
            return self.character_meanings[tamil_word]

        # Try partial matches
        for tamil_key, english_val in self.tamil_english_dict.items():
            if tamil_key in tamil_word:
                return english_val

        # Transliterate if no match found
        return self.transliterate(tamil_word)
    
    def transliterate_direct(self, tamil_text: str) -> str:
        """
        Direct transliteration from Tamil to Roman script
        Uses comprehensive character mapping for accurate phonetic conversion

        Args:
            tamil_text: Tamil text to transliterate

        Returns:
            Transliterated Roman text
        """
        result = ""
        i = 0
        while i < len(tamil_text):
            char = tamil_text[i]

            # Check if it's a consonant followed by a vowel sign
            if i + 1 < len(tamil_text) and char in self.tamil_to_roman:
                next_char = tamil_text[i + 1]

                # If next char is a vowel sign (matra)
                if next_char in ['ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ']:
                    # Get consonant base (remove inherent 'a')
                    consonant_base = self.tamil_to_roman[char]
                    if consonant_base.endswith('a'):
                        consonant_base = consonant_base[:-1]

                    # Add the vowel sound
                    vowel_sound = self.tamil_to_roman[next_char]
                    result += consonant_base + vowel_sound
                    i += 2
                    continue

            # Regular character mapping
            if char in self.tamil_to_roman:
                result += self.tamil_to_roman[char]
            else:
                result += char

            i += 1

        return result if result else "[unknown]"

    def transliterate(self, tamil_text: str) -> str:
        """
        Transliterate Tamil text to English (phonetic)

        Args:
            tamil_text: Tamil text to transliterate

        Returns:
            Transliterated English text
        """
        result = ""
        for char in tamil_text:
            if char in self.char_patterns:
                result += self.char_patterns[char]
            else:
                result += char
        return result if result else "[unknown]"
    
    def recognize_words_from_characters(self, tamil_text: str) -> List[str]:
        """
        Try to recognize complete words from continuous Tamil text
        by matching against dictionary keys

        Args:
            tamil_text: Continuous Tamil text

        Returns:
            List of recognized words
        """
        words = []
        i = 0

        while i < len(tamil_text):
            # Try to match longest word first (greedy matching)
            matched = False

            # Try matching from longest to shortest
            for length in range(min(10, len(tamil_text) - i), 0, -1):
                substring = tamil_text[i:i+length]
                if substring in self.tamil_english_dict:
                    words.append(substring)
                    i += length
                    matched = True
                    break

            if not matched:
                # If no word match, take single character
                words.append(tamil_text[i])
                i += 1

        return words

    def translate_text_direct(self, tamil_text: str) -> Dict:
        """
        Direct transliteration of Tamil text to Roman script
        Converts Tamil characters directly to their Roman phonetic equivalents

        Args:
            tamil_text: Tamil text to transliterate

        Returns:
            Dictionary with transliteration and metadata
        """
        # Direct transliteration
        roman_text = self.transliterate_direct(tamil_text)

        return {
            'tamil_text': tamil_text,
            'roman_text': roman_text,
            'english_text': roman_text,
            'method': 'direct_transliteration',
            'total_characters': len(tamil_text),
            'confidence': 100.0
        }

    def translate_text(self, tamil_text: str, split_by_character: bool = False, use_direct_transliteration: bool = False) -> Dict:
        """
        Translate Tamil text to English

        Args:
            tamil_text: Tamil text to translate
            split_by_character: If True, use word recognition instead of character splitting
            use_direct_transliteration: If True, use direct Tamil-to-Roman transliteration

        Returns:
            Dictionary with translation and metadata
        """
        # Use direct transliteration if requested
        if use_direct_transliteration:
            return self.translate_text_direct(tamil_text)

        # Determine how to split text
        if ' ' in tamil_text:
            # Space-separated words
            words = tamil_text.split()
        elif split_by_character:
            # Try to recognize words from continuous text
            words = self.recognize_words_from_characters(tamil_text)
        else:
            # Split by individual characters
            words = list(tamil_text)

        translated_words = []
        word_mappings = []

        for word in words:
            english_word = self.translate_word(word)
            translated_words.append(english_word)
            # Check if word is in dictionary or character meanings
            found_in_dict = (word in self.tamil_english_dict) or (word in self.character_meanings)
            word_mappings.append({
                'tamil': word,
                'english': english_word,
                'found_in_dict': found_in_dict
            })

        # Combine translated words
        english_text = ' '.join(translated_words)

        result = {
            'tamil_text': tamil_text,
            'english_text': english_text,
            'word_mappings': word_mappings,
            'total_words': len(words),
            'dictionary_matches': sum(1 for m in word_mappings if m['found_in_dict']),
            'transliterated_words': sum(1 for m in word_mappings if not m['found_in_dict']),
            'confidence': (sum(1 for m in word_mappings if m['found_in_dict']) / len(words) * 100) if words else 0
        }

        return result
    
    def translate_character_sequence(self, char_sequence: List[str], 
                                    tamil_text: str) -> Dict:
        """
        Translate character sequence with Tamil text
        
        Args:
            char_sequence: List of character names
            tamil_text: Corresponding Tamil Unicode text
            
        Returns:
            Dictionary with translation and metadata
        """
        translation = self.translate_text(tamil_text)
        
        result = {
            'character_sequence': char_sequence,
            'tamil_text': tamil_text,
            'english_text': translation['english_text'],
            'word_mappings': translation['word_mappings'],
            'total_words': translation['total_words'],
            'dictionary_matches': translation['dictionary_matches'],
            'transliterated_words': translation['transliterated_words'],
            'confidence': translation['confidence'],
            'status': 'success'
        }
        
        return result
    
    def get_word_info(self, tamil_word: str) -> Dict:
        """
        Get detailed information about a Tamil word
        
        Args:
            tamil_word: Tamil word to look up
            
        Returns:
            Dictionary with word information
        """
        if tamil_word in self.tamil_english_dict:
            return {
                'tamil': tamil_word,
                'english': self.tamil_english_dict[tamil_word],
                'found': True,
                'type': 'dictionary'
            }
        else:
            return {
                'tamil': tamil_word,
                'english': self.transliterate(tamil_word),
                'found': False,
                'type': 'transliterated'
            }
    
    def get_word_literal_meaning(self, tamil_word: str) -> Dict:
        """
        Get literal meaning of a Tamil word with detailed explanation

        Args:
            tamil_word: Tamil word to look up

        Returns:
            Dictionary with word meaning, category, and explanation
        """
        tamil_word = tamil_word.strip()

        if tamil_word in self.tamil_word_meanings:
            meaning_data = self.tamil_word_meanings[tamil_word]
            return {
                'tamil': tamil_word,
                'english': meaning_data['english'],
                'category': meaning_data['category'],
                'meaning': meaning_data['meaning'],
                'explanation': meaning_data['explanation'],
                'usage': meaning_data['usage'],
                'found': True,
                'type': 'literal_meaning'
            }
        elif tamil_word in self.tamil_english_dict:
            return {
                'tamil': tamil_word,
                'english': self.tamil_english_dict[tamil_word],
                'category': 'unknown',
                'meaning': self.tamil_english_dict[tamil_word],
                'explanation': f'Translation: {self.tamil_english_dict[tamil_word]}',
                'usage': 'common',
                'found': True,
                'type': 'dictionary'
            }
        else:
            return {
                'tamil': tamil_word,
                'english': self.transliterate_direct(tamil_word),
                'category': 'unknown',
                'meaning': 'unknown',
                'explanation': f'Transliteration: {self.transliterate_direct(tamil_word)}',
                'usage': 'unknown',
                'found': False,
                'type': 'transliterated'
            }

    def translate_with_literal_meaning(self, tamil_text: str) -> Dict:
        """
        Translate Tamil text with literal meanings and explanations
        Provides word-by-word breakdown with meanings and grammar info

        Args:
            tamil_text: Tamil text to translate

        Returns:
            Dictionary with literal translation and detailed word meanings
        """
        # Split text into words
        if ' ' in tamil_text:
            words = tamil_text.split()
        else:
            # Use word recognition to identify complete words from continuous text
            words = self.recognize_words_from_characters(tamil_text)

        word_meanings = []
        english_words = []

        for word in words:
            word_info = self.get_word_literal_meaning(word)
            word_meanings.append(word_info)
            english_words.append(word_info['english'])

        # Combine English words into a sentence
        english_text = ' '.join(english_words)

        # Calculate confidence based on found words
        found_count = sum(1 for w in word_meanings if w['found'])
        confidence = (found_count / len(word_meanings) * 100) if word_meanings else 0

        return {
            'tamil_text': tamil_text,
            'english_text': english_text,
            'word_meanings': word_meanings,
            'total_words': len(words),
            'found_words': found_count,
            'transliterated_words': len(words) - found_count,
            'confidence': confidence,
            'method': 'literal_meaning',
            'status': 'success'
        }

    def format_translation(self, result: Dict) -> str:
        """
        Format translation result as readable string
        
        Args:
            result: Translation result dictionary
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 70)
        output.append("TAMIL TO ENGLISH TRANSLATION")
        output.append("=" * 70)
        output.append(f"\nTamil Text: {result['tamil_text']}")
        output.append(f"English Text: {result['english_text']}")
        output.append(f"\nStatistics:")
        output.append(f"  • Total Words: {result['total_words']}")
        output.append(f"  • Dictionary Matches: {result['dictionary_matches']}")
        output.append(f"  • Transliterated: {result['transliterated_words']}")
        output.append(f"  • Confidence: {result['confidence']:.1f}%")
        
        if result['word_mappings']:
            output.append(f"\nWord-by-Word Mapping:")
            for mapping in result['word_mappings']:
                match_type = "✓" if mapping['found_in_dict'] else "~"
                output.append(f"  {match_type} {mapping['tamil']} → {mapping['english']}")
        
        output.append("\n" + "=" * 70)
        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Tamil to English Translator - Example Usage")
    print("=" * 70)
    
    # Initialize translator
    translator = TamilEnglishTranslator()
    
    # Example 1: Single word
    print("\n[Example 1] Single word translation")
    tamil_word = "கால"
    result = translator.get_word_info(tamil_word)
    print(f"Tamil: {result['tamil']}")
    print(f"English: {result['english']}")
    print(f"Type: {result['type']}")
    
    # Example 2: Text translation
    print("\n[Example 2] Text translation")
    tamil_text = "கால கல் கர்"
    result = translator.translate_text(tamil_text)
    print(translator.format_translation(result))

