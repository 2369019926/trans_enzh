# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry
import tensorflow as tf

ENZH_BPE_DATASETS = {
    "TRAIN": "train_bpe_enzh",
    "DEV": "aval_bpe_enzh"
}


def get_enzh_bpe_dataset(directory, filename):
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".en") and
            tf.gfile.Exists(train_path + ".zh")):
        raise Exception("there should be some training/dev data in the tmp dir.")

    return train_path


@registry.register_problem
class TranslateEnzhBpe32000(translate.TranslateProblem):
    """根据英德和英中的问题修改而来，这里是将英德的一个单词表变为中英的两个单词表来进行数据生成。"""

    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def source_vocab_name(self):
        return "vocab.bpe.en.%d" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.bpe.zh.%d" % self.approx_vocab_size

    def get_vocab(self, data_dir, is_target=False):
        """返回的是一个encoder，单词表对应的编码器"""
        vocab_filename = os.path.join(data_dir, self.target_vocab_name if is_target else self.source_vocab_name)
        if not tf.gfile.Exists(vocab_filename):
            raise ValueError("Vocab %s not found" % vocab_filename)
        return text_encoder.TokenTextEncoder(vocab_filename, replace_oov="UNK")

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Instance of token generator for the WMT en->zh task, training set."""
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = (ENZH_BPE_DATASETS["TRAIN"] if train else ENZH_BPE_DATASETS["DEV"])
        train_path = get_enzh_bpe_dataset(tmp_dir, dataset_path)

        # Vocab
        src_token_path = (os.path.join(data_dir, self.source_vocab_name), self.source_vocab_name)
        tar_token_path = (os.path.join(data_dir, self.target_vocab_name), self.target_vocab_name)
        for token_path, vocab_name in [src_token_path, tar_token_path]:
            if not tf.gfile.Exists(token_path):
                token_tmp_path = os.path.join(tmp_dir, vocab_name)
                tf.gfile.Copy(token_tmp_path, token_path)
                with tf.gfile.GFile(token_path, mode="r") as f:
                    vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
                with tf.gfile.GFile(token_path, mode="w") as f:
                    f.write(vocab_data)

        return text_problems.text2text_txt_iterator(train_path + ".en",
                                                    train_path + ".zh")

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """在生成数据的时候，主要是通过这个方法获取已编码样本的"""
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_vocab(data_dir)
        target_encoder = self.get_vocab(data_dir, is_target=True)
        return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                        has_inputs=self.has_inputs)

    def feature_encoders(self, data_dir):
        source_token = self.get_vocab(data_dir)
        target_token = self.get_vocab(data_dir, is_target=True)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

