#!/usr/bin/env python3
"""
Tamil Sentence Generator
Converts recognized character sequences into meaningful Tamil text/sentences
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple

class TamilSentenceGenerator:
    """
    Generates Tamil sentences from recognized character sequences
    """
    
    def __init__(self):
        """Initialize with Tamil character mappings and dictionary"""
        
        # Character class to Tamil Unicode mapping
        self.char_mapping = {
            'a': 'அ',      # a
            'aa': 'ஆ',     # aa
            'i': 'இ',      # i
            'ii': 'ஈ',     # ii
            'u': 'உ',      # u
            'uu': 'ஊ',     # uu
            'e': 'எ',      # e
            'ee': 'ஏ',     # ee
            'ai': 'ஐ',     # ai
            'o': 'ஒ',      # o
            'oo': 'ஓ',     # oo
            'au': 'ஔ',     # au
            'k': 'க',      # ka
            'kh': 'கா',    # kha
            'ng': 'ங',     # nga
            'c': 'ச',      # cha
            'ch': 'சா',    # chha
            'ny': 'ஞ',     # nya
            't': 'ட',      # ta
            'th': 'டா',    # tha
            'n': 'ண',      # na
            'p': 'ப',      # pa
            'ph': 'பா',    # pha
            'm': 'ம',      # ma
            'y': 'ய',      # ya
            'r': 'ர',      # ra
            'l': 'ல',      # la
            'v': 'வ',      # va
            'sh': 'ஷ',     # sha
            's': 'ஸ',      # sa
            'h': 'ஹ',      # ha
            'l5': 'ள',     # lla
            'r5': 'ற',     # rra
            'n1': 'ன',     # nna
            'n2': 'ந',     # nha
            'n3': 'ண',     # nna (alt)
            'n5': 'ன',     # nna (alt)
            'pu4': 'ப',    # pa (variant)
            'r5i': 'றி',   # ri
            'ru': 'ரு',    # ru
            'l5u': 'ளு',   # llu
            'l5u4': 'ளு',  # llu (variant)
            'n1u4': 'னு',  # nnu
            'n2u4': 'நு',  # nhu
            'ti': 'தி',    # thi
        }
        
        # Common Tamil words (for demonstration)
        self.tamil_dictionary = {
            'கால': {'meaning': 'Time/Leg', 'type': 'noun'},
            'கன்': {'meaning': 'Dense', 'type': 'adjective'},
            'கல்': {'meaning': 'Stone', 'type': 'noun'},
            'கர்': {'meaning': 'Hand', 'type': 'noun'},
            'கவி': {'meaning': 'Poet', 'type': 'noun'},
            'கை': {'meaning': 'Hand', 'type': 'noun'},
            'கோ': {'meaning': 'Cow', 'type': 'noun'},
            'சக': {'meaning': 'Friend', 'type': 'noun'},
            'சல': {'meaning': 'Water', 'type': 'noun'},
            'சாக': {'meaning': 'Proof', 'type': 'noun'},
            'சாய': {'meaning': 'Lean', 'type': 'verb'},
            'சிறு': {'meaning': 'Small', 'type': 'adjective'},
            'சுக': {'meaning': 'Happiness', 'type': 'noun'},
            'சுவை': {'meaning': 'Taste', 'type': 'noun'},
            'தக': {'meaning': 'Suitable', 'type': 'adjective'},
            'தல': {'meaning': 'Place', 'type': 'noun'},
            'தாய்': {'meaning': 'Mother', 'type': 'noun'},
            'தி': {'meaning': 'Day', 'type': 'noun'},
            'தீ': {'meaning': 'Fire', 'type': 'noun'},
            'தெய்': {'meaning': 'God', 'type': 'noun'},
            'தொல்': {'meaning': 'Ancient', 'type': 'adjective'},
            'நக': {'meaning': 'Nail', 'type': 'noun'},
            'நல': {'meaning': 'Good', 'type': 'adjective'},
            'நாய்': {'meaning': 'Dog', 'type': 'noun'},
            'நீ': {'meaning': 'You', 'type': 'pronoun'},
            'நீர்': {'meaning': 'Water', 'type': 'noun'},
            'பக': {'meaning': 'Side', 'type': 'noun'},
            'பல': {'meaning': 'Many', 'type': 'adjective'},
            'பாய்': {'meaning': 'Run', 'type': 'verb'},
            'பாவ': {'meaning': 'Sin', 'type': 'noun'},
            'பி': {'meaning': 'After', 'type': 'preposition'},
            'பொல': {'meaning': 'Lie', 'type': 'noun'},
            'மக': {'meaning': 'Son', 'type': 'noun'},
            'மல': {'meaning': 'Flower', 'type': 'noun'},
            'மாய': {'meaning': 'Illusion', 'type': 'noun'},
            'மி': {'meaning': 'Small', 'type': 'adjective'},
            'முக': {'meaning': 'Face', 'type': 'noun'},
            'முல': {'meaning': 'First', 'type': 'adjective'},
            'யக': {'meaning': 'Yoke', 'type': 'noun'},
            'ரக': {'meaning': 'Taste', 'type': 'noun'},
            'ரல': {'meaning': 'Sound', 'type': 'noun'},
            'ரா': {'meaning': 'Night', 'type': 'noun'},
            'ரி': {'meaning': 'Debt', 'type': 'noun'},
            'வக': {'meaning': 'Class', 'type': 'noun'},
            'வல': {'meaning': 'Strength', 'type': 'noun'},
            'வாய்': {'meaning': 'Mouth', 'type': 'noun'},
            'வி': {'meaning': 'Spread', 'type': 'verb'},
            'வீ': {'meaning': 'House', 'type': 'noun'},
            'வெய்': {'meaning': 'Sun', 'type': 'noun'},
            'வொல': {'meaning': 'Word', 'type': 'noun'},
        }
        
        print("✓ Tamil Sentence Generator initialized")
        print(f"✓ Loaded {len(self.char_mapping)} character mappings")
        print(f"✓ Loaded {len(self.tamil_dictionary)} Tamil words")
    
    def characters_to_tamil(self, char_classes: List[str]) -> str:
        """
        Convert recognized character classes to Tamil Unicode text
        
        Args:
            char_classes: List of recognized character class names
            
        Returns:
            Tamil Unicode string
        """
        tamil_text = ""
        for char_class in char_classes:
            if char_class in self.char_mapping:
                tamil_text += self.char_mapping[char_class]
            else:
                tamil_text += f"[{char_class}]"  # Unknown character
        
        return tamil_text
    
    def find_words(self, tamil_text: str) -> List[Tuple[str, Dict]]:
        """
        Find known Tamil words in the text
        
        Args:
            tamil_text: Tamil Unicode string
            
        Returns:
            List of (word, metadata) tuples
        """
        found_words = []
        
        for word, metadata in self.tamil_dictionary.items():
            if word in tamil_text:
                found_words.append((word, metadata))
        
        return found_words
    
    def generate_sentence(self, char_classes: List[str], 
                         confidence_scores: List[float] = None) -> Dict:
        """
        Generate a complete sentence from recognized characters
        
        Args:
            char_classes: List of recognized character class names
            confidence_scores: Optional list of confidence scores
            
        Returns:
            Dictionary with sentence, words, and metadata
        """
        
        # Convert to Tamil text
        tamil_text = self.characters_to_tamil(char_classes)
        
        # Find known words
        found_words = self.find_words(tamil_text)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        result = {
            "character_sequence": char_classes,
            "tamil_text": tamil_text,
            "found_words": [
                {
                    "word": word,
                    "meaning": metadata.get("meaning", "Unknown"),
                    "type": metadata.get("type", "Unknown")
                }
                for word, metadata in found_words
            ],
            "average_confidence": float(avg_confidence),
            "confidence_scores": confidence_scores if confidence_scores else [],
            "status": "success" if tamil_text else "failed"
        }
        
        return result
    
    def format_output(self, result: Dict) -> str:
        """
        Format the result as a readable string
        
        Args:
            result: Result dictionary from generate_sentence
            
        Returns:
            Formatted string output
        """
        output = []
        output.append("=" * 70)
        output.append("TAMIL SENTENCE GENERATION RESULT")
        output.append("=" * 70)
        output.append(f"\nCharacter Sequence: {' → '.join(result['character_sequence'])}")
        output.append(f"Tamil Text: {result['tamil_text']}")
        output.append(f"Average Confidence: {result['average_confidence']*100:.2f}%")
        
        if result['found_words']:
            output.append("\nRecognized Words:")
            for word_info in result['found_words']:
                output.append(f"  • {word_info['word']}: {word_info['meaning']} ({word_info['type']})")
        else:
            output.append("\nNo recognized words found in dictionary")
        
        output.append("\n" + "=" * 70)
        
        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "=" * 70)
    print("Tamil Sentence Generator - Example Usage")
    print("=" * 70)
    
    # Initialize generator
    generator = TamilSentenceGenerator()
    
    # Example 1: Simple character sequence
    print("\n[Example 1] Simple character sequence")
    char_sequence_1 = ['k', 'a', 'l']
    confidence_1 = [0.95, 0.88, 0.92]
    
    result_1 = generator.generate_sentence(char_sequence_1, confidence_1)
    print(generator.format_output(result_1))
    
    # Example 2: Another sequence
    print("\n[Example 2] Another character sequence")
    char_sequence_2 = ['n', 'a', 'l']
    confidence_2 = [0.90, 0.85, 0.88]
    
    result_2 = generator.generate_sentence(char_sequence_2, confidence_2)
    print(generator.format_output(result_2))
    
    # Save example output
    print("\nSaving example output to JSON...")
    with open('sentence_generation_example.json', 'w', encoding='utf-8') as f:
        json.dump({
            "example_1": result_1,
            "example_2": result_2
        }, f, ensure_ascii=False, indent=2)
    print("✓ Saved to: sentence_generation_example.json")

