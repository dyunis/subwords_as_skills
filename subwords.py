# from https://huggingface.co/course/chapter6/5?fw=pt
from collections import defaultdict, OrderedDict
import multiprocessing
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import numpy as np
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)
from tqdm import tqdm


def get_vocab(config, traj_dataset):
    assert config.k_actions is not None
    actions = [a.tolist() for a in traj_dataset['actions']]
    # can't remove tokens if we want to be faithful and use huggingface
    # for processing dataset for BC, but very huge action-space is hard
    if config.prune_vocab:
        vocab_size = config.max_vocab_size
    else:
        vocab_size = config.vocab_size

    vocab, tokenizer = huggingface_tokenize_trajectories(traj_dataset['actions'], config.max_vocab_size, mode=config.tokenizer)
    if config.prune_vocab:
        subwords_by_len = defaultdict(lambda: [])
        for i in range(min(config.max_vocab_size, len(vocab))):
            subwords_by_len[len(vocab[i])].append(vocab[i])

        # take first config.vocab_size subwords with the desired length (these were merged first)
        # then next length down, and so on
        subword_len = config.max_subword_len
        vocab = []
        while len(vocab) < config.vocab_size:
            vocab_left = config.vocab_size - len(vocab)
            to_add = subwords_by_len[subword_len][:vocab_left]
            vocab.extend(to_add)
            subword_len -= 1
        assert len(vocab) == config.vocab_size  # choose large enough # of merges to ensure this

    if config.k_actions == 'random':
        vocab = random_k_actions_like(vocab, config.seed)
    elif config.k_actions == 'repeated':
        vocab = repeated_k_actions_like(vocab)

    return vocab, tokenizer


def to_str(int_list):
    return ''.join([chr(i) for i in int_list])


def random_k_actions_like(vocab, seed=0):
    rng = np.random.default_rng(seed)
    # get primitives from vocab
    nums = [c for v in vocab.values() for c in v]
    primitives = sorted(list(set(nums)))
    compounds = [v for v in vocab.values() if len(v) > 1]
    # create new list of compound actions like vocab
    random_k_actions = [
        rng.choice(primitives, len(seq)).tolist() for seq in compounds
    ]
    new_vocab = [[p] for p in primitives]
    new_vocab.extend(random_k_actions)
    return new_vocab


def repeated_k_actions_like(vocab):
    # get primitives from vocab
    nums = [c for v in vocab.values() for c in v]
    primitives = sorted(list(set(nums)))
    compounds = [v for v in vocab.values() if len(v) > 1]
    # create new list of compound actions like vocab
    repeated_k_actions = [
        [seq[0] for _ in seq] for seq in compounds
    ]
    new_vocab = [[p] for p in primitives]
    new_vocab.extend(repeated_k_actions)
    return new_vocab


def huggingface_tokenize_trajectories(trajectories, vocab_size, mode='bpe'):
    # trajectories is list of lists of ints, convert to strings for huggingface
    # map to ascii first so that all primitives are preserved
    strings = [''.join([chr(c) for c in t]) for t in trajectories]
    def get_training_corpus():
        for i in range(0, len(strings), 1000):
            yield strings[i: i+1000]

    if mode == 'bpe':
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size)
    elif mode == 'wordpiece':
        tokenizer = Tokenizer(models.WordPiece())
        # funny error if unk isn't provided here
        # wordpiece isn't a faithful tokenizer, how to deal with to allow offline RL?
        # trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=['[UNK]'])
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size)
    elif mode == 'unigram':
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size)
    else:
        raise ValueError(f'Mode {mode} must be one of [bpe, wordpiece, unigram]')
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    vocab = tokenizer.get_vocab()  # chars -> int
    vocab = {v: k for k, v in vocab.items()}  # int -> chars
    if mode == 'wordpiece':
        # strip double hashes which denote subword tokens
        vocab = {k: v[2:] if v.startswith('##') else v for k, v in vocab.items()}
    vocab = {k: [ord(s) for s in v] for k, v in vocab.items()}  # int -> list[int]
    return vocab, tokenizer


if __name__ == '__main__':
    # for unigram, first make BPE corpus with very large vocab, then apply unigram on top
    # digits pre-tokenizer

    # yields batches of 1000
    import random
    def get_training_corpus():
        for i in range(100):
            yield [str(random.getrandbits(100)) for _ in range(1000)]

    tokenizer = Tokenizer(models.BPE())
    # tokenizer.pre_tokenizer = pre_tokenizers.Digits(True)
    # tokens = tokenizer.pre_tokenizer.pre_tokenize_str('123')
    trainer = trainers.BpeTrainer(vocab_size=64)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    # vocab dictionary, subword str -> ix
    vocab = tokenizer.get_vocab()
    vocab = list(vocab.keys())
    # convert to List[List[Int]]
    vocab = [[int(d) for d in list(v)] for v in vocab]
