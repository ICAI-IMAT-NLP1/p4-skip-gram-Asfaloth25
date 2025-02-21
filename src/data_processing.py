from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file

    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.

    sort_func = lambda val, d: (d[val], val)
    # changed the typing for convenience
    sorted_vocab: List[Tuple[int, str]] = [
        sort_func(val, word_counts) for val in word_counts
    ]
    
    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {
        i: val[1] for i, val in enumerate(sorted_vocab)
    }
    vocab_to_int: Dict[str, int] = {
        int_to_vocab[k]: k for k in int_to_vocab
    }

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO

    # removing the words if they are not in the vocabulary
    words = list(filter(lambda x: x in vocab_to_int, words))

    # Convert words to integers
    
    # I made this a generator so as not to flood the memory
    int_words: List[int] = list(map(vocab_to_int.get, words))
    
    counter_words = Counter(words)
    n_words : int = len(words)
    freqs: Dict[str, float] = {
        k: counter_words[k]/n_words for k in counter_words
    }

    prob_wi : Dict[str, float] = {
        k: 1 - (threshold/freqs[k])**1/2 for k in freqs
    }

    int_words_tensor : torch.Tensor = torch.tensor(
        int_words, dtype=int
    )

    prob_out : torch.Tensor = torch.tensor(
        [prob_wi[n] for n in words]
    )

    words_in : torch.Tensor = int_words_tensor[
        torch.rand_like(prob_out) > prob_out
    ]

    train_words: List[int] = words_in.tolist()

    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    target_words: List[str] = list(filter(lambda x: x != words[idx], words[max(idx-window_size,0):idx+window_size]))

    return target_words

def get_batches(words: List[int], batch_size: int, window_size: int = 5):
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    # TODO
    for idx in range(0, len(words), batch_size):
        targets = get_target(words, idx, window_size)
        inputs = [words[idx] for _ in targets]
        yield inputs, targets

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO

    valid_examples: torch.Tensor = torch.randint(0, valid_window, (valid_size,), device=device)
    valid_vecs = embedding(valid_examples)

    vecs_norms = valid_vecs / valid_vecs.norm(dim=1, keepdim=True)

    weights : torch.Tensor = embedding.weight
    embedding_norms :torch.Tensor = weights / weights.norm(dim=1, keepdim=True)
    similarities: torch.Tensor = vecs_norms @ embedding_norms.T

    return valid_examples, similarities