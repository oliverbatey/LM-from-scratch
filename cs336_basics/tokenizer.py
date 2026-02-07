import heapq
import os
import cProfile
import regex as re

from collections import defaultdict, Counter
from typing import BinaryIO, Callable, Iterable


class TokenNode:
    def __init__(self, symbol_id: int, seq_id: int):
        self.symbol_id = symbol_id  # This is the id of the TokenNode in the vocabulary.
        self.seq_id = seq_id  # An id for the TokenSequence that the TokenNode belongs to.
        self.next = None
        self.prev = None
        self.alive = True


class TokenSequence:
    def __init__(self, seq_id: int):
        """Initialise a TokenSequence with sentinel TokenNodes"""
        self.head: TokenNode = TokenNode(symbol_id=None, seq_id=seq_id)
        self.tail: TokenNode = TokenNode(symbol_id=None, seq_id=seq_id)
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

    def insert_at_head(self, token_node: TokenNode):
        current_first_node = self.head.next
        token_node.next = current_first_node
        token_node.prev = self.head
        self.head.next = token_node
        current_first_node.prev = token_node


    def insert_at_tail(self, token_node: TokenNode):
        current_last_node = self.tail.prev
        token_node.prev = current_last_node
        token_node.next = self.tail
        self.tail.prev = token_node
        current_last_node.next = token_node

    def _validate_left_node(self, node: TokenNode):
        assert node is not None
        assert node.alive is True
        assert node is not self.head and node is not self.tail
        assert node.prev is not None and node.next is not None

    def _validate_right_node(self, node: TokenNode):
        assert node is not None
        assert node.alive is True
        assert node is not self.tail
        assert node.next is not None

    def _sever_node(self, node: TokenNode):
        node.alive = False
        node.next, node.prev = None, None

    def merge_at(self, left: TokenNode, symbol_id: int) -> tuple:
        """Merge pairs of TokenNode using the left node as a reference

        Merges the sequence X <-> A <-> B <-> Y into X <-> C <-> Y
        """
        self._validate_left_node(left)
        A = left
        X = A.prev
        B = A.next
        self._validate_right_node(B)
        Y = B.next
        C = TokenNode(symbol_id=symbol_id, seq_id=self.seq_id,)
        X.next = C
        C.prev = X
        C.next = Y
        Y.prev = C
        self._sever_node(A)
        self._sever_node(B)
        return X, C, Y

    @classmethod
    def from_tokens(cls, seq_id: int, tokens: Iterable[int], node_factory: Callable):
        sequence = cls(seq_id=seq_id)
        for token in tokens:
            node = node_factory(symbol_id=token, seq_id=seq_id)
            sequence.insert_at_tail(node)
        return sequence

    def symbol_counts(self):
        symbol_counts = defaultdict(int)
        for token_node in self:
            if token_node.symbol_id:
                symbol_counts[token_node.symbol_id] +=1
        return symbol_counts

    def total_node_weights(self):
        return sum(self.symbol_counts().values())



class TokenSequenceRegister:
    def __init__(self, pretoken_counts: dict[tuple[int, ...], int], special_tokens: list[str]):
        self.pretoken_counts = pretoken_counts
        self.sequence_weights: dict[int, int] = dict()
        self.sequence_tokens: dict[int, TokenSequence] = dict()
        self.pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.pair_occurrences: dict[tuple[int, int], list[TokenNode]] = defaultdict(list)
        self.pair_count_heap: list[tuple[int, tuple[bytes, bytes], tuple[int, int]]] = []
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = initialise_vocab(special_tokens)
        self.build_sequence_registry()

    def _create_token_node(self, symbol_id: int, seq_id, **kwargs) -> TokenNode:
        return TokenNode(symbol_id=symbol_id, seq_id=seq_id)

    def _build_sequence_registry(self):
        for seq_id, sequence in enumerate(self.pretoken_counts):
            self.sequence_weights[seq_id] = self.pretoken_counts[sequence]
            self.sequence_tokens[seq_id] = TokenSequence.from_tokens(
                seq_id=seq_id, tokens=sequence, node_factory=self._create_token_node
            )

    def _build_pair_counts(self):
        for k, v in self.pretoken_counts.items():
            for j in range(len(k) - 1):
                pair = (k[j], k[j + 1])
                self.pair_counts[pair] += v

    def _build_pair_occurances(self):
        for seq_id, token_seq in self.sequence_tokens.items():
            for token in token_seq:
                if (token.next is not token_seq.tail) and (token.alive and token.next.alive):
                    self.pair_occurrences[(token.symbol_id, token.next.symbol_id)].append(token)

    def _build_pair_count_heap(self):
        self.pair_counts = {(97, 98): 10, (97, 99): 10, (97, 111): 10, (55, 11): 2}
        for pair, pair_count in self.pair_counts.items():

            self.pair_count_heap.append(
                (
                    -1 * pair_count,
                    (TokenSequenceRegister._invert_bytes(self.vocab[pair[0]]), TokenSequenceRegister._invert_bytes(self.vocab[pair[1]])),
                     pair
                )
            )
        heapq.heapify(self.pair_count_heap)

    def get_max_byte_pair(self):
        neg_count, _, pair = heapq.heappop(self.pair_count_heap)
        current_count = self.pair_counts[pair]
        while -1*neg_count != current_count and current_count > 0:  # Checks for stale counts on the heap
            neg_count, tie_key, pair = heapq.heappop(self.pair_count_heap)
        return pair

    @staticmethod
    def _invert_bytes(bytearray: bytes) -> bytes:
        return bytes([255 - b for b in bytearray])

    @staticmethod
    def is_valid_occurrence_handle(left: Optional[TokenNode], target_pair: tuple[int, int]) -> bool:
        if left is None or not left.alive or left.symbol_id is None:
            return False
        right = left.next
        if right is None or not right.alive or right.symbol_id is None:
            return False
        return (left.symbol_id, right.symbol_id) == target_pair

    # TODO: update validation to expect linked list structure
    def _validate_build_sequence_registry(self):
        assert len(self.pretoken_counts) == len(self.sequence_tokens), "Registry has a different number of elements than the pretoken counts"

    def _validate_build_pair_counts(self):
        ...

    def _validate_build_pair_occurances(self):
        ...

    def _validate(self):
        self._validate_build_sequence_registry()
        self._validate_build_pair_counts()
        self._validate_build_pair_occurances()

    def build_sequence_registry(self):
        self._build_sequence_registry()
        self._build_pair_counts()
        self._build_pair_occurances()
        self._build_pair_count_heap()
        self._validate()


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


if __name__ == "__main__":
    #path = "data/TinyStoriesV2-GPT4-valid.txt"
    path = "cs336_basics/assets/smallcorpus.txt"
    vocab_size=1000
    special_tokens = ["<|endoftext|>"]
    pretoken_counts = build_pretoken_counts(path=path, special_tokens=special_tokens)
    register = TokenSequenceRegister(pretoken_counts, special_tokens)
    sequence_weights = register.sequence_weights
    sequence_tokens = register.sequence_tokens
    pair_counts = register.pair_counts
    pair_occurrences = register.pair_occurrences


    breakpoint()
    # cProfile.run(
    #     "train_bpe(path, vocab_size, special_tokens)",
    #     sort="cumtime",
    # )