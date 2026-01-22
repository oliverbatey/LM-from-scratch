import os
import cProfile
import regex as re

from collections import defaultdict, Counter
from typing import BinaryIO, Iterable

class TokenSequenceRegister:
    def __init__(self, pretoken_counts: dict[tuple[int, ...], int]):
        self.pretoken_counts = pretoken_counts
        self.sequence_weights: dict[int, int] = dict()
        self.sequence_tokens: dict[int, tuple[int, ...]] = dict()

    def _build_sequence_registry(self):
        for seq_id, sequence in enumerate(self.pretoken_counts):
            self.sequence_weights[seq_id] = self.pretoken_counts[sequence]
            self.sequence_tokens[seq_id] = sequence

    def _validate_registry(self):
        assert len(self.pretoken_counts) == len(self.sequence_weights), "Registry has a different number of elements than the pretoken counts"
        assert sum(self.sequence_weights.values()) == sum(self.pretoken_counts.values())
        for seq_id, sequence in self.sequence_tokens.items():
            assert self.sequence_weights[seq_id] == self.pretoken_counts[sequence]

    def build_sequence_registry(self):
        self._build_sequence_registry()
        self._validate_registry()

class TokenSequence:
    def __init__(self, seq_id: int):
        """Initialise a TokenSequence with sentinel TokenNodes"""
        self.head: TokenNode = TokenNode(symbol_id=None, node_id=None, seq_id=seq_id)
        self.tail: TokenNode = TokenNode(symbol_id=None, node_id=None, seq_id=seq_id)
        self.head.prev=None
        self.head.next=self.tail
        self.tail.prev=self.head
        self.tail.next=None
        self.seq_id = seq_id

    def __iter__(self):
        s = self.head.next
        while s is not self.tail:
            yield s
            s = s.next

    def insert_at_head(self, s: TokenNode):
        new_node = TokenNode
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node

    def insert_at_tail(self, s: TokenNode):
        ...

    def merge_at(self, s: TokenNode, new_symbol: TokenNode):
        ...

    @classmethod
    def from_tokens(self, seq_id: int, tokens: Iterable[int]):
        ...

class TokenNode:
    def __init__(self, node_id: int, symbol_id: int, seq_id: int):
        self.node_id = node_id  # Unique id for the node.
        self.symbol_id = symbol_id  # This is the id of the TokenNode in the vocabulary.
        self.seq_id = seq_id  # An id for the TokenSequence that the TokenNode belongs to.
        self.next = None
        self.prev = None
        self.alive = True


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def initialise_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Initialize the BPE vocabulary with single bytes and special tokens.

    Creates a base vocabulary mapping token IDs 0-255 to their corresponding
    single-byte values, then appends special tokens starting at ID 256.

    Args:
        special_tokens: List of special token strings (e.g., ["<|endoftext|>"]).

    Returns:
        A dictionary mapping token IDs to their byte representations.
    """
    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    return vocab


def get_pretoken_counts(text: str) -> dict[tuple[int, ...], int]:
    """
    Pre-tokenize text using GPT-2 regex pattern and count occurrences.

    Splits text into pre-tokens (words, contractions, numbers, punctuation, whitespace)
    using the GPT-2 tokenization pattern. Each pre-token is encoded as a tuple of
    byte values (integers 0-255).

    Args:
        text: The input text to pre-tokenize.

    Returns:
        A dictionary mapping pre-token byte tuples to their occurrence counts.
        Keys are tuples of integers (byte values), values are counts.
    """
    pretoken_counts = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, text):
        pretoken_counts[tuple(match.group(0).encode("utf-8"))] += 1
    return pretoken_counts


def merge(t: tuple[int, ...], pair: tuple[int, int], new_index: int) -> tuple[int, ...]:
    """
    Merge all occurrences of a byte pair in a token sequence.

    Replaces every adjacent (pair[0], pair[1]) in the tuple with new_index.

    Args:
        t: A tuple of token IDs representing a pre-token.
        pair: The byte pair (two adjacent token IDs) to merge.
        new_index: The new token ID to replace the merged pair with.

    Returns:
        A new tuple with all occurrences of the pair merged into new_index.

    Example:
        >>> merge((104, 101, 108, 108, 111), (108, 108), 256)
        (104, 101, 256, 111)
    """
    seen = set()
    new_key = []
    i = 0
    token_sequence_length = len(t)
    while i < token_sequence_length:
        if (
            i + 1 < token_sequence_length and (t[i], t[i + 1]) == pair
        ):  # i+1<len(t) checks next element exists
            new_key.append(new_index)
            i += 2  # If current and the next element is merged, skip to element after next.
        else:
            new_key.append(t[i])
            i += 1
    return tuple(new_key)


def update_pretoken_counts(
    pretoken_counts: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_index: int,
) -> dict[tuple[int, ...], int]:
    """
    Apply a merge operation to all pre-tokens and aggregate counts.

    For each pre-token in the counts dictionary, merges the given byte pair
    and combines counts for any pre-tokens that become identical after merging.

    Args:
        pretoken_counts: Dictionary mapping pre-token tuples to their counts.
        pair: The byte pair to merge in all pre-tokens.
        new_index: The new token ID for the merged pair.

    Returns:
        Updated dictionary with merged pre-tokens and aggregated counts.
    """
    d = defaultdict(int)
    for key, value in pretoken_counts.items():
        key = merge(key, pair, new_index)
        d[key] += value
    return d

def build_pretoken_counts(
    path: str, special_tokens: list[str], num_processes: int = 4
) -> dict[tuple[int, ...], int]:
    """
    Build pre-token counts from a text file, splitting on special tokens.

    Reads the file in chunks, splits each chunk on special tokens (known to be document delimiters) to avoid
    learning merges across document boundaries, then counts all pre-tokens using the GPT-2 regex pattern.

    Args:
        path: Path to the input text file.
        special_tokens: List of special tokens to split on (e.g., ["<|endoftext|>"]).
        num_processes: Number of chunks to split the file into.

    Returns:
        A dictionary mapping pre-token byte tuples to their total counts.
    """
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        pretoken_counts = defaultdict(int)
        split_pattern = "|".join([re.escape(t) for t in special_tokens])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for subchunk in re.split(split_pattern, chunk):
                for pretoken, count in get_pretoken_counts(subchunk).items():
                    pretoken_counts[pretoken] += count
    return pretoken_counts

def train(
    pretoken_counts: dict[tuple[int, ...], int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE merges from pre-computed pre-token counts.

    Starting from a base vocabulary of 256 single bytes plus special tokens,
    iteratively finds the most frequent adjacent byte pair and merges them
    until the desired vocabulary size is reached.

    Args:
        pretoken_counts: Dictionary mapping pre-token byte tuples to counts,
            as returned by build_pretoken_counts().
        vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens)).
        special_tokens: List of special token strings to include in the vocabulary.

    Returns:
        A tuple containing:
        - vocab: Dictionary mapping token IDs to their byte representations.
        - merges: List of (bytes, bytes) tuples representing each merge operation,
          where each tuple contains the byte representations of the two tokens
          that were merged, in the order they were merged.
    """
    merges: list[tuple[bytes, bytes]] = []
    vocab = initialise_vocab(special_tokens)
    new_index = max(vocab.keys())
    while len(vocab) < vocab_size:
        byte_pair_count = defaultdict(int)
        new_index += 1
        for k, v in pretoken_counts.items():
            for j in range(len(k) - 1):
                pair = (k[j], k[j + 1])
                byte_pair_count[pair] += v

        most_frequent_byte_pair = max(
            # Tie-breaking with the lexographically greatest pair in bytes representation, NOT integer token ID representation.
            byte_pair_count, key=lambda key: (byte_pair_count[key], vocab[key[0]], vocab[key[1]])
        )
        merges.append((vocab[most_frequent_byte_pair[0]], vocab[most_frequent_byte_pair[1]]))
        vocab[new_index] = (
            vocab[most_frequent_byte_pair[0]] + vocab[most_frequent_byte_pair[1]]
        )
        pretoken_counts = update_pretoken_counts(
            pretoken_counts, most_frequent_byte_pair, new_index
        )
    return vocab, merges

def train_bpe(
    path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on a text corpus.

    Starting from a base vocabulary of 256 single bytes plus special tokens,
    iteratively finds the most frequent adjacent byte pair across all pre-tokens
    and merges them into a new token until the desired vocabulary size is reached.

    Args:
        path: Path to the training text file.
        vocab_size: Target vocabulary size (must be >= 256 + len(special_tokens)).
        special_tokens: List of special token strings to include in the vocabulary.

    Returns:
        A tuple containing:
        - vocab: Dictionary mapping token IDs to their byte representations.
        - merges: List of (token_id_1, token_id_2) tuples in the order they were merged.
    """
    pretoken_counts = build_pretoken_counts(path, special_tokens)
    return train(pretoken_counts, vocab_size, special_tokens)


def build_sequence_registry(
    pretoken_counts: dict[tuple[int, ...], int]
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    sequence_weight = defaultdict(int)
    sequence_tokens = dict()
    for seq_id, sequence in enumerate(pretoken_counts):
        sequence_weight[seq_id] += pretoken_counts[sequence]
        sequence_tokens[seq_id] = sequence
    return sequence_weight, sequence_tokens





if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size=1000
    special_tokens = ["<|endoftext|>"]
    pretoken_counts = build_pretoken_counts(path=path, special_tokens=special_tokens)
    register = TokenSequenceRegister(pretoken_counts)
    register.build_sequence_registry()
    sequence_weights = register.sequence_weights
    sequence_tokens = register.sequence_tokens

    cProfile.run(
        "train_bpe(path, vocab_size, special_tokens)",
        sort="cumtime",
    )