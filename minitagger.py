# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module contains the code to train and use Minitagger.
"""
import argparse
import collections
import math
import numpy
import os
import pickle
import subprocess
import sys

# Specify where to find liblinear.
liblinear_path = os.path.join(os.path.dirname(__file__),
                              "liblinear-1.94/python")
sys.path.append(os.path.abspath(liblinear_path))
import liblinearutil

############################# code about data ##################################
class SequenceData(object):
    """
    Represents a dataset of sequences. The sequences can be partially labeled.
    They can be loaded from a text file or a list.
    """

    def __init__(self, given_data):
        self.data_path = None
        self.sequence_pairs = []
        self.num_instances = 0
        self.num_labeled_instances = 0
        self.observation_count = collections.Counter()
        self.label_count = collections.Counter()
        self.observation_label_count = {}  # "set" => {"verb":10, "noun":8}

        if isinstance(given_data, str):
            self.__initialize_sequence_pairs_from_file(given_data)
        elif isinstance(given_data, list):
            self.__initialize_sequence_pairs_from_list(given_data)
        else:
            raise Exception("A sequence data can be constructed from either a "
                            "string (file path) or a list")
        self.__initialize_attributes()

    def __initialize_sequence_pairs_from_file(self, data_path):
        """
        Initializes sequences from a text file. The format is:
        [observation] [optional: label]
        Empty lines indicate sequence boundaries.
        """
        self.data_path = data_path
        with open(data_path, "r") as infile:
            observation_sequence = []
            label_sequence = []
            def append_pair():
                """Appends the current sequence pair to self.sequence_pairs."""
                self.sequence_pairs.append([observation_sequence,
                                            label_sequence])
            for line in infile:
                toks = line.split()
                assert len(toks) < 3
                if toks:
                    observation = toks[0]
                    label = None if len(toks) == 1 else toks[1]
                    observation_sequence.append(observation)
                    label_sequence.append(label)
                else:
                    if observation_sequence:
                        append_pair()
                        observation_sequence = []
                        label_sequence = []

            if observation_sequence:
                append_pair()

    def __initialize_sequence_pairs_from_list(self, sequence_list):
        """
        Initializes sequences from the given list. The i-th element of the given
        list should have the following form:
        sequence_list[i] = [observation_sequence, label_sequence]
        A label absence is denoted with None.
        """
        for sequence_pair in sequence_list:
            assert len(sequence_pair) == 2
            observation_sequence = sequence_pair[0]
            label_sequence = sequence_pair[1]
            assert len(observation_sequence) == len(label_sequence)
            self.sequence_pairs.append([observation_sequence, label_sequence])

    def __initialize_attributes(self):
        """
        Initializes the dataset attributes from the loaded sequences.
        """
        for observation_sequence, label_sequence in self.sequence_pairs:
            for i in range(len(observation_sequence)):
                observation = observation_sequence[i]
                self.num_instances += 1
                self.observation_count[observation] += 1

                label = label_sequence[i]
                if label is not None:
                    self.num_labeled_instances += 1
                    self.label_count[label] += 1
                    if not observation in self.observation_label_count:
                        self.observation_label_count[observation] = \
                            collections.Counter()
                        self.observation_label_count[observation][label] += 1

        for observation in self.observation_label_count:
            self.observation_label_count[observation] = sorted(
                self.observation_label_count[observation].items(),
                key=lambda pair: pair[1], reverse=True)

############################# code about features ##############################
FRONT_BUFFER_SYMBOL = "_START_"
END_BUFFER_SYMBOL = "_END_"
UNKNOWN_SYMBOL = "<?>"

def merge_dicts(*dicts):
    """Merges the given dictionaries into one."""
    return dict(chain(*[d.iteritems() for d in dicts]))

def get_word(word_sequence, position):
    """Gets the word at the specified position."""
    if position < 0:
        return FRONT_BUFFER_SYMBOL
    elif position >= len(word_sequence):
        return END_BUFFER_SYMBOL
    else:
        return word_sequence[position]

def is_capitalized(word):
    """Is the word capitalized?"""
    return word[0].isupper()

def get_prefix(word, length):
    """Gets a padded prefix of the word up to the given length."""
    prefix = ""
    for i in range(length):
        if i < len(word):
            prefix += word[i]
        else:
            prefix += "*"
    return prefix

def get_suffix(word, length):
    """Gets a padded suffix of the word up to the given length."""
    suffix = ""
    for i in range(length):
        if i < len(word):
            suffix = word[-i-1] + suffix
        else:
            suffix = "*" + suffix
    return suffix

def is_all_nonalphanumeric(word):
    """Is the word all nonalphanumeric?"""
    for char in word:
        if char.isalnum():
            return False
    return True

def is_float(word):
    """Can the word be converted to a float (i.e., numeric value)?"""
    try:
        float(word)
        return True
    except:
        return False

# FEATURE_CACHE[(word, relative_position)] stores the features extracted for the
# word at the relative position so that the features can be immediate retrieved
# if requested again.
global FEATURE_CACHE
FEATURE_CACHE = {}

def clear_feature_cache():
    """Clears the global feature cache."""
    global FEATURE_CACHE
    FEATURE_CACHE = {}

def spelling_features(word, relative_position):
    """
    Extracts spelling features about the given word. Also considers the word's
    relative position.
    """
    if not (word, relative_position) in FEATURE_CACHE:
        features = {}
        features["word(0)={1}".format(relative_position, word)] = 1
        features['is_capitalized({0})={1}'.format(
                relative_position, is_capitalized(word))] = 1
        for length in range(1, 5):
            features["prefix{0}({1})={2}".format(
                    length, relative_position, get_prefix(word, length))] = 1
            features["suffix{0}({1})={2}".format(
                    length, relative_position, get_suffix(word, length))] = 1
        features["is_all_nonalphanumeric({0})={1}".format(
                relative_position, is_all_nonalphanumeric(word))] = 1
        features["is_float({0})={1}".format(
                relative_position, is_float(word))] = 1
        FEATURE_CACHE[(word, relative_position)] = features

    # Return a copy so that modifying that object doesn't modify the cache.
    return FEATURE_CACHE[(word, relative_position)].copy()

def get_baseline_features(word_sequence, position):
    """
    Baseline features: spelling of the word at the position, identities of
    2 words left and right of the word.
    """
    word = get_word(word_sequence, position)
    word_left1 = get_word(word_sequence, position - 1)
    word_left2 = get_word(word_sequence, position - 2)
    word_right1 = get_word(word_sequence, position + 1)
    word_right2 = get_word(word_sequence, position + 2)

    features = spelling_features(word, 0)
    features["word(-1)={0}".format(word_left1)] = 1
    features["word(-2)={0}".format(word_left2)] = 1
    features["word(+1)={0}".format(word_right1)] = 1
    features["word(+2)={0}".format(word_right1)] = 1
    return features

def get_embedding_features(word_sequence, position, embedding_dictionary):
    """
    Embedding features: normalized baseline features + (normalized) embeddings
    of current, left, and right words.
    """
    # Compute the baseline feature vector and normalize its length to 1.
    features = get_baseline_features(word_sequence, position)
    norm_features = math.sqrt(len(features))  # Assumes binary feature values
    for feature in features:
        features[feature] /= norm_features

    # Add the (already normalized) embedding features.
    word = word_sequence[position]  # current word
    if word in embedding_dictionary:
        word_embedding = embedding_dictionary[word]
    else:
        word_embedding = embedding_dictionary[UNKNOWN_SYMBOL]
    for i, value in enumerate(word_embedding):
        features["embedding(0)_at({0})".format(i + 1)] = value

    if position > 0:
        word = word_sequence[position - 1]  # word to the left
        if word in embedding_dictionary:
            word_embedding = embedding_dictionary[word]
        else:
            word_embedding = embedding_dictionary[UNKNOWN_SYMBOL]
        for i, value in enumerate(word_embedding):
            features["embedding(-1)_at({0})".format(i + 1)] = value

    if position < len(word_sequence) - 1:
        word = word_sequence[position + 1]  # word to the right
        if word in embedding_dictionary:
            word_embedding = embedding_dictionary[word]
        else:
            word_embedding = embedding_dictionary[UNKNOWN_SYMBOL]
        for i, value in enumerate(word_embedding):
            features["embedding(+1)_at({0})".format(i + 1)] = value

    return features

def get_bitstring_features(word_sequence, position, bitstring_dictionary):
    """
    Bit string features: baseline features + bit strings of current, left, and
    right words.
    """
    # Compute the baseline feature vector.
    features = get_baseline_features(word_sequence, position)

    # Add the bit string features.
    word = word_sequence[position]  # current word
    if word in bitstring_dictionary:
        word_bitstring = bitstring_dictionary[word]
    else:
        word_bitstring = bitstring_dictionary[UNKNOWN_SYMBOL]
    for i in range(1, len(word_bitstring) + 1):
        features["bitstring(0)_prefix({0})={1}".format(
                i, word_bitstring[:i])] = 1
    features["bitstring(0)_all={0}".format(word_bitstring)] = 1

    if position > 0:
        word = word_sequence[position - 1]  # word to the left
        if word in bitstring_dictionary:
            word_bitstring = bitstring_dictionary[word]
        else:
            word_bitstring = bitstring_dictionary[UNKNOWN_SYMBOL]
        for i in range(1, len(word_bitstring) + 1):
            features["bitstring(-1)_prefix({0})={1}".format(
                    i, word_bitstring[:i])] = 1
        features["bitstring(-1)_all={0}".format(word_bitstring)] = 1

    if position < len(word_sequence) - 1:
        word = word_sequence[position + 1]  # word to the right
        if word in bitstring_dictionary:
            word_bitstring = bitstring_dictionary[word]
        else:
            word_bitstring = bitstring_dictionary[UNKNOWN_SYMBOL]
        for i in range(1, len(word_bitstring) + 1):
            features["bitstring(+1)_prefix({0})={1}".format(
                    i, word_bitstring[:i])] = 1
        features["bitstring(+1)_all={0}".format(word_bitstring)] = 1

    return features

class SequenceDataFeatureExtractor():
    """Extracts features from sequence data."""

    def __init__(self, feature_template):
        clear_feature_cache()
        self.feature_template = feature_template
        self.data_path = None
        self.is_training = True
        self.__map_feature_str2num = {}
        self.__map_feature_num2str = {}
        self.__map_label_str2num = {}
        self.__map_label_num2str = {}
        self.__word_embedding = None
        self.__word_bitstring = None

    def num_feature_types(self):
        return len(self.__map_feature_str2num)

    def get_feature_string(self, feature_number):
        assert feature_number in self.__map_feature_num2str
        return self.__map_feature_num2str[feature_number]

    def get_label_string(self, label_number):
        assert label_number in self.__map_label_num2str
        return self.__map_label_num2str[label_number]

    def get_feature_number(self, feature_string):
        assert feature_string in self.__map_feature_str2num
        return self.__map_feature_str2num[feature_string]

    def get_label_number(self, label_string):
        assert label_string in self.__map_label_str2num
        return self.__map_label_str2num[label_string]

    def extract_features(self, sequence_data):
        """Extracts features from the given sequence data."""
        label_list = []
        features_list = []

        self.data_path = sequence_data.data_path
        for (observation_sequence, label_sequence) in \
                sequence_data.sequence_pairs:
            for i in range(len(observation_sequence)):
                if label_sequence[i] is not None:  # Only use labeled instances.
                    label_list.append(self.__get_label(label_sequence[i]))
                    features_list.append(
                        self.__get_features(observation_sequence, i))

        return label_list, features_list

    def __get_label(self, label):
        """Returns the integer ID of the given label."""
        if self.is_training:
            # If training, add unknown label types to the dictionary.
            if not label in self.__map_label_str2num:
                label_number = len(self.__map_label_str2num)
                self.__map_label_str2num[label] = label_number
                self.__map_label_num2str[label_number] = label
            return self.__map_label_str2num[label]
        else:
            # If predicting, consult the trained dictionary.
            if label in self.__map_label_str2num:
                return self.__map_label_str2num[label]
            else:
                return len(self.__map_label_str2num)  # Unknown label

    def __get_features(self, observation_sequence, i):
        """
        Returns the integer IDs of the extracted features for the i-th
        position of the given observation sequence.
        """
        # Extract raw features.
        if self.feature_template == "baseline":
            raw_features = get_baseline_features(observation_sequence, i)
        elif self.feature_template == "embedding":
            assert self.__word_embedding is not None
            raw_features = get_embedding_features(observation_sequence, i,
                                                  self.__word_embedding)
        elif self.feature_template == "bitstring":
            assert self.__word_bitstring is not None
            raw_features = get_bitstring_features(observation_sequence, i,
                                                  self.__word_bitstring)
        else:
            raise Exception("Unsupported feature template {0}".format(
                    self.feature_template))

        # Convert raw features into integer IDs.
        numeric_features = {}
        for raw_feature in raw_features:
            if self.is_training:
                # If training, add unknown feature types to the dictionary.
                if not raw_feature in self.__map_feature_str2num:
                    feature_number = len(self.__map_feature_str2num)
                    self.__map_feature_str2num[raw_feature] = feature_number
                    self.__map_feature_num2str[feature_number] = raw_feature
                numeric_features[self.__map_feature_str2num[raw_feature]] = \
                    raw_features[raw_feature]
            else:
                # if predicting, only consider known feature types.
                if raw_feature in self.__map_feature_str2num:
                    numeric_features[self.__map_feature_str2num[raw_feature]] \
                        = raw_features[raw_feature]
        return numeric_features

    def load_word_embeddings(self, embedding_path):
        self.__word_embedding = {}
        with open(embedding_path, "r") as infile:
            for line in infile:
                toks = line.split()
                if len(toks) == 0: continue

                # toks = [count] [type] [value_1] ... [value_m]
                self.__word_embedding[toks[1]] = \
                    numpy.array([float(tok) for tok in toks[2:]])

                # Always normalize word embeddings.
                self.__word_embedding[toks[1]] /= \
                    numpy.linalg.norm(self.__word_embedding[toks[1]])

        # Assert that the token for unknown word types is present.
        assert UNKNOWN_SYMBOL in self.__word_embedding

    def load_word_bitstrings(self, bitstring_path):
        self.__word_bitstring = {}
        with open(bitstring_path, "r") as infile:
            for line in infile:
                toks = line.split()
                if len(toks) == 0: continue

                # toks = [bitstring] [type] [count]
                self.__word_bitstring[toks[1]] = toks[0]

        # Assert that the token for unknown word types is present.
        assert UNKNOWN_SYMBOL in self.__word_bitstring

############################# code about model  ################################
class Minitagger():
    """Main tagger model"""

    def __init__(self):
        self.__feature_extractor = None
        self.__liblinear_model = None
        self.quiet = False

    def equip_feature_extractor(self, feature_extractor):
        self.__feature_extractor = feature_extractor

    def train(self, data_train):
        """Trains Minitagger on the given data."""
        assert self.__feature_extractor.is_training  # Assert untrained

        # Extract features and pass them to liblinear.
        [label_list, features_list] = \
            self.__feature_extractor.extract_features(data_train)
        if not self.quiet:
            print("{0} labeled instances (out of {1})".format(
                    data_train.num_labeled_instances, data_train.num_instances))
            print("{0} label types".format(len(data_train.label_count)))
            print("{0} observation types".format(
                    len(data_train.observation_count)))
            print("\"{0}\" feature template".format(
                    self.__feature_extractor.feature_template))
            print("{0} feature types".format(
                    self.__feature_extractor.num_feature_types()))
        problem = liblinearutil.problem(label_list, features_list)
        self.__liblinear_model = \
            liblinearutil.train(problem, liblinearutil.parameter("-q"))
        self.__feature_extractor.is_training = False

    def save(self, model_path):
        if not os.path.isdir(model_path):
            if os.path.exists(model_path):
                subprocess.check_output(["rm", "-rf", model_path])
            os.makedirs(model_path)
        pickle.dump(self.__feature_extractor,
                    open(os.path.join(model_path, "feature_extractor"), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        liblinearutil.save_model(os.path.join(model_path, "liblinear_model"),
                                 self.__liblinear_model)

    def load(self, model_path):
        self.__feature_extractor = pickle.load(
            open(os.path.join(model_path, "feature_extractor"), "rb"))
        self.__liblinear_model = liblinearutil.load_model(
            os.path.join(model_path, "liblinear_model"))

    def predict(self, data_test):
        assert not self.__feature_extractor.is_training  # Assert trained

        # Extract features and pass them to liblinear for prediction.
        [label_list, features_list] = \
            self.__feature_extractor.extract_features(data_test)
        pred_labels, (acc, _, _), _ = \
            liblinearutil.predict(label_list, features_list,
                                  self.__liblinear_model, "-q")
        if not self.quiet:
            print("Per-instance accuracy: {0:.3f}%".format(acc))
        return pred_labels, acc

######################## script for command line usage  ########################
def main(args):
    """Runs the main function."""
    minitagger = Minitagger()
    minitagger.quiet = args.quiet
    sequence_data = SequenceData(args.data_path)  # Given data

    if args.train:
        # Train on that data.
        feature_extractor = SequenceDataFeatureExtractor(args.feature_template)
        if args.embedding_path:
            feature_extractor.load_word_embeddings(args.embedding_path)
        if args.bitstring_path:
            feature_extractor.load_word_bitstrings(args.bitstring_path)
        minitagger.equip_feature_extractor(feature_extractor)
        minitagger.train(sequence_data)
        minitagger.save(args.model_path)

    else:
        # Predict tags in that data.
        minitagger = Minitagger()
        minitagger.load(args.model_path)
        pred_labels, acc = minitagger.predict(sequence_data)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_path", type=str, help="path to model "
                           "directory")
    argparser.add_argument("data_path", type=str, help="path to data (used for "
                           "training/testing)")
    argparser.add_argument("--train", action="store_true", help="train the "
                           "tagger on the given data")
    argparser.add_argument("--feature_template",
                           type=str, default="baseline", help="feature template"
                           " (default %(default)s)")
    argparser.add_argument("--embedding_path", type=str, help="path to word "
                           "embeddings")
    argparser.add_argument("--bitstring_path", type=str, help="path to word "
                           "bit strings (from a hierarchy of word types)")
    argparser.add_argument("--quiet", action="store_true", help="no messages")
    parsed_args = argparser.parse_args()
    main(parsed_args)
