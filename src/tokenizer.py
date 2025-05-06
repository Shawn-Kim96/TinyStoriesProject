import collections
import re

class BPETokenizerOptimized:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = []                              # list of pairs merged in order
        self.merge_regex = {}                         # for fast encoding
        self.vocab = set()                            # final token vocab

    def train(self, corpus):
        # 1) Build initial word→freq mapping, where each word is a tuple of chars + '</w>'
        word_freq = collections.Counter(
            tuple(word) + ('</w>',)
            for sentence in corpus
            for word in sentence.strip().split()
        )

        # 2) Compute initial pair frequencies
        pair_freq = self._get_pair_stats(word_freq)

        # 3) Iteratively perform merges
        for _ in range(self.num_merges):
            if not pair_freq:
                break
            best_pair, _ = max(pair_freq.items(), key=lambda kv: kv[1])
            self.merges.append(best_pair)

            # 4) Merge that pair in all words *and* update frequencies incrementally
            word_freq, pair_freq = self._merge_pair(best_pair, word_freq, pair_freq)

        # 5) Build final vocab
        for word in word_freq:
            self.vocab.update(word)

        # 6) Precompile regex patterns for fast encoding
        self._build_regex()

    def _get_pair_stats(self, word_freq):
        stats = collections.Counter()
        for word, freq in word_freq.items():
            for a, b in zip(word[:-1], word[1:]):
                stats[(a, b)] += freq
        return stats

    def _merge_pair(self, pair, word_freq, old_stats):
        a, b = pair
        new_word_freq = collections.Counter()
        new_stats = collections.Counter()

        # We’ll only recompute counts for pairs affected by this merge
        for word, freq in word_freq.items():
            # merge occurrences of (a,b) → ab
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            merged = tuple(merged)
            new_word_freq[merged] += freq

        # recompute all pair stats (for simplicity, but you could also update incrementally)
        new_stats = self._get_pair_stats(new_word_freq)
        return new_word_freq, new_stats

    def _build_regex(self):
        # build a pattern that matches any of our merges, e.g. ('a','b') → 'ab'
        # we compile one regex per merge, in order, for deterministic encoding
        for a, b in self.merges:
            pat = re.escape(a) + r'\s+' + re.escape(b)
            self.merge_regex[(a, b)] = re.compile(pat)

    def encode(self, token):
        # start with chars + end-of-word marker
        symbols = token + '</w>'
        # apply merges in learned order
        for pair in self.merges:
            pattern = self.merge_regex[pair]
            # join symbols into string separated by spaces for regex
            s = ' '.join(symbols)
            if pattern.search(s):
                s = pattern.sub(''.join(pair), s)
                symbols = s.split()
        return symbols

    def encode_sentence(self, sentence):
        out = []
        for word in sentence.strip().split():
            out.extend(self.encode(word))
        return out

    def get_vocab(self):
        return self.vocab
