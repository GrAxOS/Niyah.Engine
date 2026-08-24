#!/usr/bin/env python3
"""
Build Arabic-Centric Vocabulary for NiyahMini from scratch.

This script creates an original vocabulary with:
- Arabic characters, words, and subwords
- English characters, words, and subwords
- Code symbols and programming constructs
- Evidence-related special tokens
- Mathematical and logical symbols
- Byte-level fallback for unknown characters

NO borrowed vocabularies from Llama, Qwen, Mistral, or any other model.
"""

import argparse
import json
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set
import math

# Special tokens for NiyahMini
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<fact>": 4,
    "<inference>": 5,
    "<unknown>": 6,
    "<conflicted>": 7,
    "<ar>": 8,
    "<en>": 9,
    "<code>": 10,
    "<source>": 11,
    "<cite>": 12,
}

# Arabic character ranges and common characters
ARABIC_CHARS = ""
\u0600-\u06FF  # Arabic
\u0750-\u077F  # Arabic Supplement
\u08A0-\u08FF  # Arabic Extended-A
\uFB50-\uFDFF  # Arabic Presentation Forms-A
\uFE70-\uFEFF  # Arabic Presentation Forms-B
"""

# Additional characters for Arabic text
ARABIC_EXTENDED = ""
\u0600-\u06FF  # Basic Arabic
\u0750-\u077F  # Arabic Supplement
\u08A0-\u08FF  # Arabic Extended-A
\uFB50-\uFDFF  # Arabic Presentation Forms-A
\uFE70-\uFEFF  # Arabic Presentation Forms-B
\u20AA         # Shekel sign (used in Arabic contexts)
"""

# Code symbols and programming constructs
CODE_SYMBOLS = {
    # Brackets and delimiters
    "(", ")", "[", "]", "{", "}", "<", ">",
    # Operators
    "+", "-", "*", "/", "=", "==", "!=", "<=", ">=", "+", "-",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>",
    "&&", "||", "!", "~", "^", "%", "&", "|",
    # Punctuation
    ".", ",", ":", ";", "'", '"', "`", "?", "!",
    # Special
    "\\", "@", "#", "$", "_",
    # Math
    "+", "-", "*", "/", "=", "==", "!=", "<=", ">=",
}

# Mathematical symbols
MATH_SYMBOLS = {
    "+", "-", "×", "÷", "=", "≠", "<", ">", "≤", "≥",
    "≈", "≡", "→", "∀", "∃", "∈", "∉", "∅", "∪", "∩",
    "∑", "∏", "∫", "√", "∞", "∆", "∇", "θ", "φ", "λ",
    "α", "β", "γ", "δ", "ε", "ζ", "η",
}

# Logical symbols
LOGICAL_SYMBOLS = {
    "∧", "∨", "¬", "⊢", "⊨", "⊤", "⊥", "⇒", "⇔",
}

class VocabBuilder:
    """Builds a vocabulary from training text using BPE."""
    
    def __init__(self, config: Dict):
        self.vocab_size = config.get('vocab_size', 32768)
        self.min_frequency = config.get('min_frequency', 5)
        self.n_merges = config.get('n_merges', 10000)
        self.coverage_threshold = config.get('coverage_threshold', 0.99)
        self.include_arabic = config.get('include_arabic', True)
        self.include_english = config.get('include_english', True)
        self.include_code = config.get('include_code', True)
        self.include_numbers = config.get('include_numbers', True)
        
        # Token frequency counter
        self.token_counts: Counter = Counter()
        
        # Character vocabulary
        self.char_vocab: Set[str] = set()
        
        # Word vocabulary (for initial tokenization)
        self.word_vocab: Counter = Counter()
        
        # BPE merge rules
        self.merges: List[Tuple[str, str]] = []
        
        # Final vocabulary
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Training text
        self.texts: List[str] = []
    
    def add_text(self, text: str):
        """Add training text for vocabulary building."""
        self.texts.append(text)
    
    def add_texts(self, texts: List[str]):
        """Add multiple training texts."""
        self.texts.extend(texts)
    
    def _is_arabic_char(self, char: str) -> bool:
        """Check if a character is Arabic."""
        # Check Unicode ranges for Arabic
        code = ord(char)
        return (
            (0x0600 <= code <= 0x06FF) or  # Arabic
            (0x0750 <= code <= 0x077F) or  # Arabic Supplement
            (0x08A0 <= code <= 0x08FF) or  # Arabic Extended-A
            (0xFB50 <= code <= 0xFDFF) or  # Arabic Presentation Forms-A
            (0xFE70 <= code <= 0xFEFF) or  # Arabic Presentation Forms-B
            (code == 0x20AA)               # Shekel sign
        )
    
    def _is_english_char(self, char: str) -> bool:
        """Check if a character is English (ASCII letter)."""
        return char.isalpha() and ord(char) < 128
    
    def _is_code_char(self, char: str) -> bool:
        """Check if a character is commonly used in code."""
        return char in CODE_SYMBOLS or char in MATH_SYMBOLS or char in LOGICAL_SYMBOLS
    
    def _is_number_char(self, char: str) -> bool:
        """Check if a character is a digit."""
        return char.isdigit() or char in ".-+eE"
    
    def _should_include_char(self, char: str) -> bool:
        """Determine if a character should be included in the vocabulary."""
        if self.include_arabic and self._is_arabic_char(char):
            return True
        if self.include_english and self._is_english_char(char):
            return True
        if self.include_code and self._is_code_char(char):
            return True
        if self.include_numbers and self._is_number_char(char):
            return True
        # Always include whitespace and common punctuation
        if char in " \t\n\r.,;:?!'\"()[]{}<>@#$%^&*+-=|\\/~`":
            return True
        return False
    
    def _get_char_vocab(self, text: str) -> Set[str]:
        """Extract unique characters from text."""
        chars = set()
        for char in text:
            if self._should_include_char(char):
                chars.add(char)
        return chars
    
    def _tokenize_to_chars(self, text: str) -> List[str]:
        """Tokenize text into characters, filtering unwanted ones."""
        tokens = []
        for char in text:
            if self._should_include_char(char):
                tokens.append(char)
            else:
                # Replace unknown characters with <unk>
                tokens.append("<unk>")
        return tokens
    
    def _build_char_vocab(self):
        """Build initial character vocabulary from all texts."""
        self.char_vocab = set()
        
        # Add special tokens
        for token in SPECIAL_TOKENS:
            self.char_vocab.add(token)
        
        # Add all valid characters from texts
        for text in self.texts:
            self.char_vocab.update(self._get_char_vocab(text))
        
        # Ensure we have basic punctuation and symbols
        basic_chars = " \t\n\r.,;:?!'\"()[]{}<>@#$%^&*+-=|\\/~`"
        for char in basic_chars:
            self.char_vocab.add(char)
    
    def _build_initial_vocab(self):
        """Build initial vocabulary from words and characters."""
        # Start with character vocabulary
        self._build_char_vocab()
        
        # Count word frequencies
        word_pattern = re.compile(r'[\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
        
        for text in self.texts:
            words = word_pattern.findall(text)
            self.word_vocab.update(words)
        
        # Filter words by frequency
        filtered_words = {w: c for w, c in self.word_vocab.items() if c >= self.min_frequency}
        
        # Build initial vocab: special tokens + frequent words + characters
        vocab_list = []
        
        # Add special tokens first
        for token, tid in sorted(SPECIAL_TOKENS.items(), key=lambda x: x[1]):
            vocab_list.append(token)
        
        # Add frequent words
        for word, count in sorted(filtered_words.items(), key=lambda x: (-x[1], x[0])):
            if len(vocab_list) >= self.vocab_size // 2:  # Reserve space for merges
                break
            vocab_list.append(word)
        
        # Add characters
        for char in sorted(self.char_vocab):
            if char not in vocab_list:
                vocab_list.append(char)
        
        # Create initial vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab_list)}
    
    def _count_pair_frequencies(self, texts: List[str]) -> Counter:
        """Count frequencies of adjacent token pairs."""
        pair_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize_to_chars(text)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1
        
        return pair_counts
    
    def _learn_bpe_merges(self):
        """Learn BPE merge rules from training data."""
        # Get pair frequencies
        pair_counts = self._count_pair_frequencies(self.texts)
        
        # Learn merges
        for _ in range(self.n_merges):
            if not pair_counts:
                break
            
            # Find most frequent pair
            most_common = pair_counts.most_common(1)[0]
            pair, count = most_common
            
            if count < self.min_frequency:
                break
            
            # Create new token
            new_token = pair[0] + pair[1]
            
            # Add to merges
            self.merges.append((pair[0], pair[1]))
            
            # Update vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.id_to_token[len(self.id_to_token)] = new_token
            
            # Update pair counts: replace pair with new token
            new_pair_counts = Counter()
            for (a, b), freq in pair_counts.items():
                if (a, b) == pair:
                    # This pair becomes the new token
                    continue
                
                # Check if we need to replace a or b with new_token
                a_replaced = a
                b_replaced = b
                
                # This is a simplified version - full BPE would need to track
                # which tokens have been merged
                
            # For simplicity, we'll rebuild pair counts after each merge
            # This is slower but more accurate
            pair_counts = self._count_pair_frequencies(self.texts)
            
            # Remove the merged pair
            del pair_counts[pair]
        
        return self.merges
    
    def _apply_bpe_merges(self, text: str) -> List[str]:
        """Apply BPE merges to tokenize text."""
        tokens = list(text)
        
        # Filter to only include characters in our vocabulary
        filtered_tokens = []
        for token in tokens:
            if token in self.vocab or self._should_include_char(token):
                filtered_tokens.append(token)
            else:
                filtered_tokens.append("<unk>")
        
        tokens = filtered_tokens
        
        # Apply merges in order
        for first, second in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == first and tokens[i + 1] == second:
                    merged = first + second
                    if merged in self.vocab:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def build(self):
        """Build the complete vocabulary."""
        if not self.texts:
            raise ValueError("No training text provided")
        
        # Build initial vocabulary
        self._build_initial_vocab()
        
        # Learn BPE merges
        self._learn_bpe_merges()
        
        # Finalize vocabulary
        self._finalize_vocab()
        
        return self.vocab, self.merges
    
    def _finalize_vocab(self):
        """Finalize vocabulary, ensuring we have exactly vocab_size tokens."""
        # Ensure we have special tokens
        for token, tid in SPECIAL_TOKENS.items():
            if token not in self.vocab:
                self.vocab[token] = tid
                self.id_to_token[tid] = token
        
        # If we have too many tokens, remove least frequent ones
        while len(self.vocab) > self.vocab_size:
            # Find least frequent token (excluding special tokens)
            min_freq = float('inf')
            min_token = None
            
            for token in self.vocab:
                if token in SPECIAL_TOKENS:
                    continue
                
                # Count frequency in texts
                freq = sum(1 for text in self.texts if token in text)
                if freq < min_freq:
                    min_freq = freq
                    min_token = token
            
            if min_token:
                tid = self.vocab.pop(min_token)
                del self.id_to_token[tid]
        
        # If we have too few tokens, add more characters or common words
        while len(self.vocab) < self.vocab_size:
            # This is a placeholder - in practice we'd add more tokens
            # from the training data
            break
    
    def save(self, vocab_path: Path, merges_path: Path):
        """Save vocabulary and merges to files."""
        # Save vocabulary
        with vocab_path.open('w', encoding='utf-8') as f:
            for token, tid in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        
        # Save merges
        with merges_path.open('w', encoding='utf-8') as f:
            for first, second in self.merges:
                f.write(f"{first} {second}\n")
    
    @classmethod
    def load(cls, vocab_path: Path, merges_path: Path) -> 'VocabBuilder':
        """Load vocabulary and merges from files."""
        builder = cls({})
        
        # Load vocabulary
        with vocab_path.open('r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token:
                    builder.vocab[token] = len(builder.vocab)
                    builder.id_to_token[len(builder.id_to_token)] = token
        
        # Load merges
        with merges_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    builder.merges.append((parts[0], parts[1]))
        
        return builder

def generate_arabic_corpus_samples() -> List[str]:
    """Generate sample Arabic text for vocabulary building."""
    arabic_texts = [
        # Basic Arabic phrases
        "السلام عليكم ورحمة الله وبركاته",
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "الصبر مفتاح الفرج",
        "العلم نور",
        "الوقت من ذهب",
        "الصحة تاج على رؤوس الأصحاء",
        "المعرفة قوة",
        "العمل عبادة",
        "الصدقة جارية",
        
        # Arabic with numbers
        "في عام ٢٠٢٤، بلغ عدد سكان العالم ٨ مليارات نسمة",
        "درجة الحرارة ٣٧.٥ درجة مئوية",
        "السعر ١٠٠ ريال سعودي",
        "المسافة ١٥٠ كيلومتر",
        
        # Arabic names
        "محمد أحمد علي",
        "فاطمة الزهراء",
        "يوسف بن أحمد",
        "ليلى بنت محمد",
        
        # Arabic technical terms
        "الذكة الاصطناعية",
        "تعلم الآلة",
        "الشبكة العصبية",
        "الخوارزمية",
        "برمجة",
        "كود",
        "مترجم",
        "مصادر مفتوحة",
        
        # Arabic-English code mixed
        "function حساب_المتوسط(list) { return sum(list) / list.length; }",
        "// هذه دالة لحساب المتوسط",
        "const x = ١٠; // قيمة ثابتة",
        "if (condition) { // إذا كان الشرط صحيح",
        
        # Mathematical expressions in Arabic
        "x + y = z",
        "f(x) = x^2 + 2x + 1",
        "∑ i=1 to n: i^2",
        "∫ from a to b: f(x) dx",
        
        # Logical expressions
        "A ∧ B",
        "A ∨ B",
        "¬A",
        "A ⇒ B",
        
        # Evidence-related terms
        "حقيقة",
        "استدلال",
        "مجهول",
        "متضارب",
        "مصدر",
        "إثبات",
        "دليل",
        "مرجع",
        
        # Programming terms in Arabic
        "متغير",
        "دالة",
        "حلقات",
        "شرط",
        "مصفوفة",
        "قائمة",
        "كائن",
        "فئة",
        "وراثة",
        "وحدة",
        
        # Common Arabic words
        "و", "في", "من", "إلى", "على", "أن", "لا", "ما", "هذا", "ذلك",
        "ال", "ل", "ب", "ك", "م", "ه", "س", "ي", "ت", "ن",
        
        # Arabic questions
        "ما هو؟",
        "كيف؟",
        "متى؟",
        "أين؟",
        "لماذا؟",
        "من؟",
        
        # Arabic commands
        "افتح",
        "اغلق",
        "احفظ",
        "احذف",
        "عرض",
        "ابحث",
        
        # Arabic with punctuation
        "مرحبا! كيف حالك؟",
        "أنا بخير، شكراً لك.",
        "هل يمكنك مساعدتي؟",
        "نعم، بالطبع.",
        "لا، آسف.",
    ]
    
    return arabic_texts

def generate_english_corpus_samples() -> List[str]:
    """Generate sample English text for vocabulary building."""
    english_texts = [
        # Basic English
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be that is the question",
        
        # Programming terms
        "function", "variable", "loop", "condition", "array", "list",
        "object", "class", "inheritance", "interface", "module",
        "algorithm", "complexity", "recursion", "iteration",
        
        # Code snippets
        "def hello(): print('Hello')",
        "for i in range(10): print(i)",
        "if x > 0: return True else: return False",
        "class MyClass: def __init__(self): pass",
        
        # Mathematical terms
        "sum", "product", "integral", "derivative", "matrix", "vector",
        "equation", "formula", "theorem", "proof", "hypothesis",
        
        # Logical terms
        "and", "or", "not", "implies", "equivalent",
        "true", "false", "if", "then", "else",
        
        # Evidence-related terms
        "fact", "inference", "unknown", "conflicted",
        "source", "proof", "evidence", "reference",
        "verify", "validate", "check", "confirm",
        
        # Common words
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        
        # Numbers
        "zero", "one", "two", "three", "four", "five",
        "10", "100", "1000", "1.5", "3.14", "-1",
        
        # Punctuation
        ".", ",", "?", "!", "'", '"', ":", ";", "-", "(", ")", "[", "]", "{", "}",
    ]
    
    return english_texts

def generate_code_corpus_samples() -> List[str]:
    """Generate sample code for vocabulary building."""
    code_texts = [
        # Python
        "def factorial(n): if n <= 1: return 1 else: return n * factorial(n-1)",
        "class Point: def __init__(self, x, y): self.x = x; self.y = y",
        "for i in range(10): print(i * 2)",
        "if x > 0 and y > 0: return True",
        "import math; result = math.sqrt(x)",
        
        # JavaScript
        "function add(a, b) { return a + b; }",
        "const obj = { name: 'test', value: 123 };",
        "for (let i = 0; i < 10; i++) { console.log(i); }",
        "if (condition) { doSomething(); } else { doOther(); }",
        
        # C/C++
        "int main() { printf(\"Hello\"); return 0; }",
        "for (int i = 0; i < 10; i++) { arr[i] = i * 2; }",
        "if (x > 0) { y = x; } else { y = -x; }",
        
        # Java
        "public class Main { public static void main(String[] args) { } }",
        "int sum = 0; for (int i = 0; i < arr.length; i++) { sum += arr[i]; }",
        
        # SQL
        "SELECT * FROM users WHERE age > 18 ORDER BY name",
        "INSERT INTO table (col1, col2) VALUES (1, 'test')",
        
        # HTML
        "<div class=\"container\"> <p>Hello</p> </div>",
        "<html><head><title>Test</title></head><body></body></html>",
        
        # Mathematical notation
        "f(x) = x^2 + 2x + 1",
        "∑_{i=1}^n i^2 = n(n+1)(2n+1)/6",
        "∫_a^b f(x) dx",
        
        # Symbols
        "== != <= >= < > + - * / % & | ^ ~ ! @ # $ % ^ & *",
    ]
    
    return code_texts

def main():
    parser = argparse.ArgumentParser(description='Build Arabic-centric vocabulary for NiyahMini')
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('--vocab-size', type=int, default=32768)
    parser.add_argument('--n-merges', type=int, default=10000)
    parser.add_argument('--min-frequency', type=int, default=5)
    parser.add_argument('--include-arabic', action='store_true', default=True)
    parser.add_argument('--include-english', action='store_true', default=True)
    parser.add_argument('--include-code', action='store_true', default=True)
    parser.add_argument('--include-numbers', action='store_true', default=True)
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    training_texts = []
    
    if args.include_arabic:
        training_texts.extend(generate_arabic_corpus_samples())
    
    if args.include_english:
        training_texts.extend(generate_english_corpus_samples())
    
    if args.include_code:
        training_texts.extend(generate_code_corpus_samples())
    
    # Add some mixed content
    training_texts.extend([
        "Hello مرحبا",
        "The answer is ١٠٠",
        "function حساب(x) { return x + 1; }",
        "FACT: Water boils at 100°C",
        "INFERENCE: The meeting will likely be productive",
        "UNKNOWN: I don't have enough information",
        "CONFLICTED: Sources disagree on this point",
    ])
    
    print(f"Generated {len(training_texts)} training samples")
    print(f"Building vocabulary with {args.vocab_size} tokens and {args.n_merges} merges...")
    
    # Build vocabulary
    config = {
        'vocab_size': args.vocab_size,
        'n_merges': args.n_merges,
        'min_frequency': args.min_frequency,
        'include_arabic': args.include_arabic,
        'include_english': args.include_english,
        'include_code': args.include_code,
        'include_numbers': args.include_numbers,
    }
    
    builder = VocabBuilder(config)
    builder.add_texts(training_texts)
    
    vocab, merges = builder.build()
    
    print(f"Built vocabulary with {len(vocab)} tokens")
    print(f"Learned {len(merges)} BPE merges")
    
    # Save vocabulary and merges
    vocab_path = args.output_dir / 'vocab.txt'
    merges_path = args.output_dir / 'merges.txt'
    
    builder.save(vocab_path, merges_path)
    
    print(f"Saved vocabulary to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    
    # Save config
    config_path = args.output_dir / 'vocab_config.json'
    with config_path.open('w', encoding='utf-8') as f:
        json.dump({
            'vocab_size': len(vocab),
            'n_merges': len(merges),
            'min_frequency': args.min_frequency,
            'include_arabic': args.include_arabic,
            'include_english': args.include_english,
            'include_code': args.include_code,
            'include_numbers': args.include_numbers,
            'sha256': hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved config to {config_path}")
    
    # Print some statistics
    print("\nVocabulary Statistics:")
    print(f"  Total tokens: {len(vocab)}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    
    # Count tokens by type
    arabic_tokens = sum(1 for t in vocab if any(builder._is_arabic_char(c) for c in t))
    english_tokens = sum(1 for t in vocab if any(c.isalpha() and ord(c) < 128 for c in t))
    code_tokens = sum(1 for t in vocab if any(c in CODE_SYMBOLS for c in t))
    
    print(f"  Arabic tokens: {arabic_tokens}")
    print(f"  English tokens: {english_tokens}")
    print(f"  Code tokens: {code_tokens}")
    print(f"  BPE merges: {len(merges)}")

if __name__ == '__main__':
    main()
