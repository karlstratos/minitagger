# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module contains the code to train and use Minitagger.
"""
import argparse
import collections
import datetime
import math
import numpy
import os
import pickle
import random
import subprocess
import sys
import time

# Specify where to find liblinear.
LIBLINEAR_PATH = os.path.join(os.path.dirname(__file__),
                              "liblinear-1.96/python")
sys.path.append(os.path.abspath(LIBLINEAR_PATH))
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
        self.is_partially_labeled = False

        if isinstance(given_data, str):
            self.__initialize_sequence_pairs_from_file(given_data)
        elif isinstance(given_data, list):
            self.__initialize_sequence_pairs_from_list(given_data)
        else:
            raise Exception("A sequence data can be constructed from either a "
                            "string (file path) or a list")
        self.__initialize_attributes()

    def get_average_length(self):
        """Calculates the average length of the sequences."""
        length_sum = 0
        for observation_sequence, _ in self.sequence_pairs:
            length_sum += len(observation_sequence)
        return float(length_sum) / len(self.sequence_pairs)

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
            for line in infile:
                toks = line.split()
                assert len(toks) < 3
                if toks:
                    observation = toks[0]
                    label = None if len(toks) == 1 else toks[1]
                    if label is None:
                        self.is_partially_labeled = True
                    observation_sequence.append(observation)
                    label_sequence.append(label)
                else:
                    if observation_sequence:
                        self.sequence_pairs.append([observation_sequence,
                                                    label_sequence])
                        observation_sequence = []
                        label_sequence = []

            if observation_sequence:
                self.sequence_pairs.append([observation_sequence,
                                            label_sequence])

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

    def __str__(self):
        """String representation of sequence pairs"""
        string_rep = ""
        for sequence_num, (observation_sequence, label_sequence) in \
                enumerate(self.sequence_pairs):
            for position in range(len(observation_sequence)):
                string_rep += observation_sequence[position]
                if not label_sequence[position] is None:
                    string_rep += "\t" + label_sequence[position]
                string_rep += "\n"
            if sequence_num < len(self.sequence_pairs) - 1:
                string_rep += "\n"
        return string_rep

def analyze_data(data_path):
    """Analyzes the given data file."""

    # Establish whether this data is a prediction file or not.
    is_prediction = False
    is_not_prediction = False
    with open(data_path, "r") as infile:
        for line in infile:
            toks = line.split()
            if toks:
                assert len(toks) < 4
                if len(toks) < 3:
                    is_not_prediction = True
                else:
                    is_prediction = True

    assert is_prediction or is_not_prediction and \
        not (is_prediction and is_not_prediction)

    # If prediction, recover the original data and also compute accuracy.
    sequence_pairs = []
    per_instance_accuracy = -1
    per_sequence_accuracy = -1
    if is_prediction:
        num_labels = 0
        num_sequences = 0
        num_correct_labels = 0
        num_correct_sequences = 0
        with open(data_path, "r") as infile:
            observation_sequence = []
            gold_label_sequence = []
            pred_label_sequence = []
            for line in infile:
                toks = line.split()
                if toks:
                    num_labels += 1
                    observation = toks[0]
                    gold_label = toks[1]
                    pred_label = toks[2]
                    if pred_label == gold_label:
                        num_correct_labels += 1
                    observation_sequence.append(observation)
                    gold_label_sequence.append(gold_label)
                    pred_label_sequence.append(pred_label)
                else:
                    num_sequences += 1
                    if observation_sequence:
                        sequence_pairs.append([observation_sequence,
                                               gold_label_sequence])
                    if pred_label_sequence == gold_label_sequence:
                        num_correct_sequences += 1
                    observation_sequence = []
                    gold_label_sequence = []
                    pred_label_sequence = []
            if observation_sequence:
                num_sequences += 1
                sequence_pairs.append([observation_sequence,
                                       gold_label_sequence])
                if pred_label_sequence is gold_label_sequence:
                    num_correct_sequences += 1
        per_instance_accuracy = float(num_correct_labels) / num_labels * 100
        per_sequence_accuracy = float(num_correct_sequences) / num_sequences \
            * 100

    # Construct sequence data.
    data = SequenceData(sequence_pairs) if is_prediction \
        else SequenceData(data_path)

    if is_prediction:
        print("A prediction data file:", data_path)
    else:
        print("A non-prediction data file:", data_path)
    print("{0} sequences (average length: {1:.1f})".format(
            len(data.sequence_pairs), data.get_average_length()))
    print("{0} instances".format(data.num_instances))
    print("{0} labeled instances".format(data.num_labeled_instances))
    print("{0} observation types".format(len(data.observation_count)))
    print("{0} label types".format(len(data.label_count)))
    if is_prediction:
        print("Per-instance accuracy: {0:.3f}%".format(per_instance_accuracy))
        print("Per-sequence accuracy: {0:.3f}%".format(per_sequence_accuracy))

############################# code about features ##############################
FRONT_BUFFER_SYMBOL = "_START_"  # For sentence boundaries
END_BUFFER_SYMBOL = "_END_"  # For sentence boundaries
UNKNOWN_SYMBOL = "<?>"   # For unknown observation types at test time

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
    except ValueError:
        return False

# FEATURE_CACHE[(word, relative_position)] stores the features extracted for the
# word at the relative position so that the features can be immediate retrieved
# if requested again.
global SPELLING_FEATURE_CACHE
SPELLING_FEATURE_CACHE = {}

def clear_spelling_feature_cache():
    """Clears the global spelling feature cache."""
    global SPELLING_FEATURE_CACHE
    SPELLING_FEATURE_CACHE = {}

def spelling_features(word, relative_position):
    """
    Extracts spelling features about the given word. Also considers the word's
    relative position.
    """
    if not (word, relative_position) in SPELLING_FEATURE_CACHE:
        features = {}
        features["word({0})={1}".format(relative_position, word)] = 1
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
        SPELLING_FEATURE_CACHE[(word, relative_position)] = features

    # Return a copy so that modifying that object doesn't modify the cache.
    return SPELLING_FEATURE_CACHE[(word, relative_position)].copy()

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
    features["word(+2)={0}".format(word_right2)] = 1
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

class SequenceDataFeatureExtractor(object):
    """Extracts features from sequence data."""

    def __init__(self, feature_template):
        clear_spelling_feature_cache()
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
        """Returns the number of distinct feature types."""
        return len(self.__map_feature_str2num)

    def get_feature_string(self, feature_number):
        """Converts a numeric feature ID to a string."""
        assert feature_number in self.__map_feature_num2str
        return self.__map_feature_num2str[feature_number]

    def get_label_string(self, label_number):
        """Converts a numeric label ID to a string."""
        assert label_number in self.__map_label_num2str
        return self.__map_label_num2str[label_number]

    def get_feature_number(self, feature_string):
        """Converts a feature string to a numeric ID."""
        assert feature_string in self.__map_feature_str2num
        return self.__map_feature_str2num[feature_string]

    def get_label_number(self, label_string):
        """Converts a label string to a numeric ID."""
        assert label_string in self.__map_label_str2num
        return self.__map_label_str2num[label_string]

    def extract_features(self, sequence_data, extract_all, skip_list):
        """
        Extracts features from the given sequence data. Also returns the
        sequence-position indices of the extracted instances. Unless specified
        extract_all=True, it extracts features only from labeled instances.

        It also skips extracting features from examples specified by skip_list.
        This is used for active learning. (Pass [] to not skip any example.)
        """
        label_list = []
        features_list = []
        location_list = []

        self.data_path = sequence_data.data_path
        for sequence_num, (observation_sequence, label_sequence) in \
                enumerate(sequence_data.sequence_pairs):
            for position, label in enumerate(label_sequence):

                # If this example is in the skip list, ignore.
                if skip_list and skip_list[sequence_num][position]:
                    continue

                # Only use labeled instances unless extract_all=True.
                if (not label is None) or extract_all:
                    label_list.append(self.__get_label(label))
                    features_list.append(
                        self.__get_features(observation_sequence, position))
                    location_list.append((sequence_num, position))

        return label_list, features_list, location_list

    def __get_label(self, label):
        """Returns the integer ID of the given label."""
        if self.is_training:
            # If training, add unknown label types to the dictionary.
            if not label in self.__map_label_str2num:
                label_number = len(self.__map_label_str2num) + 1  # index from 1
                self.__map_label_str2num[label] = label_number
                self.__map_label_num2str[label_number] = label
            return self.__map_label_str2num[label]
        else:
            # If predicting, consult the trained dictionary.
            if label in self.__map_label_str2num:
                return self.__map_label_str2num[label]
            else:
                return -1 # Unknown label

    def __get_features(self, observation_sequence, position):
        """
        Returns the integer IDs of the extracted features for observation at the
        given position in the sequence.
        """
        # Extract raw features.
        if self.feature_template == "baseline":
            raw_features = get_baseline_features(observation_sequence, position)
        elif self.feature_template == "embedding":
            assert self.__word_embedding is not None
            raw_features = get_embedding_features(observation_sequence,
                                                  position,
                                                  self.__word_embedding)
        elif self.feature_template == "bitstring":
            assert self.__word_bitstring is not None
            raw_features = get_bitstring_features(observation_sequence,
                                                  position,
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
                    # Note: Feature index has to starts from 1 in liblinear.
                    feature_number = len(self.__map_feature_str2num) + 1
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
        """Loads word embeddings from a file in the given path."""
        self.__word_embedding = {}
        with open(embedding_path, "r") as infile:
            for line in infile:
                toks = line.split()
                if len(toks) == 0:
                    continue

                # toks = [count] [type] [value_1] ... [value_m]
                self.__word_embedding[toks[1]] = \
                    numpy.array([float(tok) for tok in toks[2:]])

                # Always normalize word embeddings.
                self.__word_embedding[toks[1]] /= \
                    numpy.linalg.norm(self.__word_embedding[toks[1]])

        # Assert that the token for unknown word types is present.
        assert UNKNOWN_SYMBOL in self.__word_embedding

        # Address some treebank token conventions.
        if "(" in self.__word_embedding:
            self.__word_embedding["-LCB-"] = self.__word_embedding["("]
            self.__word_embedding["-LRB-"] = self.__word_embedding["("]
            self.__word_embedding["*LCB*"] = self.__word_embedding["("]
            self.__word_embedding["*LRB*"] = self.__word_embedding["("]
        if ")" in self.__word_embedding:
            self.__word_embedding["-RCB-"] = self.__word_embedding[")"]
            self.__word_embedding["-RRB-"] = self.__word_embedding[")"]
            self.__word_embedding["*RCB*"] = self.__word_embedding[")"]
            self.__word_embedding["*RRB*"] = self.__word_embedding[")"]
        if "\"" in self.__word_embedding:
            self.__word_embedding["``"] = self.__word_embedding["\""]
            self.__word_embedding["''"] = self.__word_embedding["\""]
            self.__word_embedding["`"] = self.__word_embedding["\""]
            self.__word_embedding["'"] = self.__word_embedding["\""]

    def load_word_bitstrings(self, bitstring_path):
        """Loads word bitstrings from a file in the given path."""
        self.__word_bitstring = {}
        with open(bitstring_path, "r") as infile:
            for line in infile:
                toks = line.split()
                if len(toks) == 0:
                    continue

                # toks = [bitstring] [type] [count]
                self.__word_bitstring[toks[1]] = toks[0]

        # Assert that the token for unknown word types is present.
        assert UNKNOWN_SYMBOL in self.__word_bitstring

        # Address some treebank token replacement conventions.
        if "(" in self.__word_bitstring:
            self.__word_bitstring["-LCB-"] = self.__word_bitstring["("]
            self.__word_bitstring["-LRB-"] = self.__word_bitstring["("]
            self.__word_bitstring["*LCB*"] = self.__word_bitstring["("]
            self.__word_bitstring["*LRB*"] = self.__word_bitstring["("]
        if ")" in self.__word_bitstring:
            self.__word_bitstring["-RCB-"] = self.__word_bitstring[")"]
            self.__word_bitstring["-RRB-"] = self.__word_bitstring[")"]
            self.__word_bitstring["*RCB*"] = self.__word_bitstring[")"]
            self.__word_bitstring["*RRB*"] = self.__word_bitstring[")"]
        if "\"" in self.__word_bitstring:
            self.__word_bitstring["``"] = self.__word_bitstring["\""]
            self.__word_bitstring["''"] = self.__word_bitstring["\""]
            self.__word_bitstring["`"] = self.__word_bitstring["\""]
            self.__word_bitstring["'"] = self.__word_bitstring["\""]

############################# code about model  ################################
class Minitagger(object):
    """Main tagger model"""

    def __init__(self):
        self.__feature_extractor = None
        self.__liblinear_model = None
        self.quiet = False
        self.active_output_path = ""
        self.active_seed_size = 0
        self.active_step_size = 0
        self.active_output_interval = 0

    def equip_feature_extractor(self, feature_extractor):
        """Equips Minitagger with a feature extractor."""
        self.__feature_extractor = feature_extractor

    def train(self, data_train, data_dev):
        """Trains Minitagger on the given data."""
        start_time = time.time()
        assert self.__feature_extractor.is_training  # Assert untrained

        # Extract features (only labeled instances) and pass them to liblinear.
        [label_list, features_list, _] = \
            self.__feature_extractor.extract_features(data_train, False, [])
        if not self.quiet:
            print("{0} labeled instances (out of {1})".format(
                    len(label_list), data_train.num_instances))
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

        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            print("Training time: {0}".format(
                    str(datetime.timedelta(seconds=num_seconds))))
            if data_dev is not None:
                quiet_value = self.quiet
                self.quiet = True
                _, acc = self.predict(data_dev)
                self.quiet = quiet_value
                print("Dev accuracy: {0:.3f}%".format(acc))

    def train_actively(self, data_train, data_dev):
        """Does margin-based active learning on the given data."""

        # We will assume that we can label every example.
        assert not data_train.is_partially_labeled

        # Keep track of which examples can be still selected for labeling.
        __skip_extraction = []
        for _, label_sequence in data_train.sequence_pairs:
            __skip_extraction.append([False for _ in label_sequence])

        # Create an output directory.
        if os.path.exists(self.active_output_path):
            subprocess.check_output(["rm", "-rf", self.active_output_path])
        os.makedirs(self.active_output_path)
        logfile = open(os.path.join(self.active_output_path, "log"), "w")

        def __make_data_from_locations(locations):
            """
            Makes SequenceData out of a subset of data_train from given
            location=(sequence_num, position) pairs.
            """
            selected_positions = collections.defaultdict(list)
            for (sequence_num, position) in locations:
                selected_positions[sequence_num].append(position)

            sequence_list = []
            for sequence_num in selected_positions:
                word_sequence, label_sequence = \
                    data_train.sequence_pairs[sequence_num]
                selected_labels = [None for _ in range(len(word_sequence))]
                for position in selected_positions[sequence_num]:
                    selected_labels[position] = label_sequence[position]

                    # This example will not be selected again.
                    __skip_extraction[sequence_num][position] = True
                sequence_list.append((word_sequence, selected_labels))

            selected_data = SequenceData(sequence_list)
            return selected_data

        def __train_silently(data_selected):
            """Trains on the argument data in silent mode."""
            self.__feature_extractor.is_training = True  # Reset for training.
            quiet_value = self.quiet
            self.quiet = True
            self.train(data_selected, None)  # No need for development here.
            self.quiet = quiet_value

        def __interval_report(data_selected):
            # Only report at each interval.
            if data_selected.num_labeled_instances % \
                    self.active_output_interval != 0:
                return

            # Test on the development data if we have it.
            if data_dev is not None:
                quiet_value = self.quiet
                self.quiet = True
                _, acc = self.predict(data_dev)
                self.quiet = quiet_value
                message = "{0} labels: {1:.3f}%".format(
                    data_selected.num_labeled_instances, acc)
                print(message)
                logfile.write(message + "\n")
                logfile.flush()

            # Output the selected labeled examples so far.
            file_name = os.path.join(
                self.active_output_path,
                "example" + str(data_selected.num_labeled_instances))
            with open(file_name, "w") as outfile:
                outfile.write(data_selected.__str__())

        # Compute the (active_seed_size) most frequent word types in data_train.
        sorted_wordcount_pairs = sorted(data_train.observation_count.items(),
                                        key=lambda type_count: type_count[1],
                                        reverse=True)
        seed_wordtypes = [wordtype for wordtype, _ in
                          sorted_wordcount_pairs[:self.active_seed_size]]

        # Select a random occurrence of each selected type for a seed example.
        occurring_locations = collections.defaultdict(list)
        for sequence_num, (observation_sequence, _) in \
                enumerate(data_train.sequence_pairs):
            for position, word in enumerate(observation_sequence):
                if word in seed_wordtypes:
                    occurring_locations[word].append((sequence_num, position))
        locations = [random.sample(occurring_locations[wordtype], 1)[0] for
                     wordtype in seed_wordtypes]
        data_selected = __make_data_from_locations(locations)
        __train_silently(data_selected)  # Train for the first time.
        __interval_report(data_selected)

        while len(locations) < data_train.num_labeled_instances:
            # Make predictions on the remaining (i.e., not on the skip list)
            # labeled examples.
            [label_list, features_list, location_list] = \
                self.__feature_extractor.extract_features(\
                data_train, False, __skip_extraction)

            _, _, scores_list = \
                liblinearutil.predict(label_list, features_list,
                                      self.__liblinear_model, "-q")

            # Compute "confidence" of each prediction:
            #   max_{y} score(x,y) - max_{y'!=argmax_{y} score(x,y)} score(x,y')
            confidence_index_pairs = []
            for index, scores in enumerate(scores_list):
                sorted_scores = sorted(scores, reverse=True)

                # Handle the binary case: liblinear gives only 1 score whose
                # sign indicates the class (+ versus -).
                confidence = sorted_scores[0] - sorted_scores[1] \
                    if len(scores) > 1 else abs(scores[0])
                confidence_index_pairs.append((confidence, index))

            # Select least confident examples for next labeling.
            confidence_index_pairs.sort()
            for _, index in confidence_index_pairs[:self.active_step_size]:
                locations.append(location_list[index])
            data_selected = __make_data_from_locations(locations)
            __train_silently(data_selected)  # Train from scratch.
            __interval_report(data_selected)

        logfile.close()

    def save(self, model_path):
        """Saves the model as a directory at the given path."""
        if os.path.exists(model_path):
            subprocess.check_output(["rm", "-rf", model_path])
        os.makedirs(model_path)
        pickle.dump(self.__feature_extractor,
                    open(os.path.join(model_path, "feature_extractor"), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        liblinearutil.save_model(os.path.join(model_path, "liblinear_model"),
                                 self.__liblinear_model)

    def load(self, model_path):
        """Loads the model from the directory at the given path."""
        self.__feature_extractor = pickle.load(
            open(os.path.join(model_path, "feature_extractor"), "rb"))
        self.__liblinear_model = liblinearutil.load_model(
            os.path.join(model_path, "liblinear_model"))

    def predict(self, data_test):
        """
        Predicts tags in the given data. If the data is fully labeled, reports
        the accuracy.
        """
        start_time = time.time()
        assert not self.__feature_extractor.is_training  # Assert trained

        # Extract features (on all instances, labeled or unlabeled) and pass
        # them to liblinear for prediction.
        [label_list, features_list, _] = \
            self.__feature_extractor.extract_features(data_test, True, [])
        pred_labels, (acc, _, _), _ = \
            liblinearutil.predict(label_list, features_list,
                                  self.__liblinear_model, "-q")
        if not self.quiet:
            num_seconds = int(math.ceil(time.time() - start_time))
            print("Prediction time: {0}".format(
                    str(datetime.timedelta(seconds=num_seconds))))
            if not data_test.is_partially_labeled:
                print("Per-instance accuracy: {0:.3f}%".format(acc))
            else:
                print("Not reporting accuracy: test data missing gold labels")

        # Convert predicted labels from integer IDs to strings.
        for i, label in enumerate(pred_labels):
            pred_labels[i] = self.__feature_extractor.get_label_string(label)
        return pred_labels, acc

######################## script for command line usage  ########################
ABSENT_GOLD_LABEL = "<NO_GOLD_LABEL>"  # Used for instances without gold labels.

def main(args):
    """Runs the main function."""

    # If specified, just analyze the given data and return. This data can be
    # a prediction output file.
    if args.analyze:
        analyze_data(args.data_path)
        return

    # Otherwise, either train or use a tagger model on the given data.
    minitagger = Minitagger()
    minitagger.quiet = args.quiet
    sequence_data = SequenceData(args.data_path)

    if args.train:
        feature_extractor = SequenceDataFeatureExtractor(args.feature_template)
        if args.embedding_path:
            feature_extractor.load_word_embeddings(args.embedding_path)
        if args.bitstring_path:
            feature_extractor.load_word_bitstrings(args.bitstring_path)
        minitagger.equip_feature_extractor(feature_extractor)
        data_dev = SequenceData(args.dev_path) if args.dev_path else None
        if data_dev is not None:  # Development data should be fully labeled.
            assert not data_dev.is_partially_labeled
        if not args.active:
            assert args.model_path
            minitagger.train(sequence_data, data_dev)
            minitagger.save(args.model_path)
        else:  # Do active learning on the training data
            assert args.active_output_path
            minitagger.active_output_path = args.active_output_path
            minitagger.active_seed_size = args.active_seed_size
            minitagger.active_step_size = args.active_step_size
            minitagger.active_output_interval = args.active_output_interval
            minitagger.train_actively(sequence_data, data_dev)

    else:  # Predict labels in the given data.
        assert args.model_path
        minitagger.load(args.model_path)
        pred_labels, _ = minitagger.predict(sequence_data)

        # Optional prediciton output.
        if args.prediction_path:
            with open(args.prediction_path, "w") as outfile:
                label_index = 0
                for sequence_num, (word_sequence, label_sequence) in \
                        enumerate(sequence_data.sequence_pairs):
                    for position, word in enumerate(word_sequence):
                        if not label_sequence[position] is None:
                            gold_label = label_sequence[position]
                        else:
                            gold_label = ABSENT_GOLD_LABEL
                        outfile.write(word + "\t" + gold_label + "\t" + \
                                          pred_labels[label_index] + "\n")
                        label_index += 1
                    if sequence_num < len(sequence_data.sequence_pairs) - 1:
                        outfile.write("\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=str, help="path to data (used for "
                           "training/testing)")
    argparser.add_argument("--analyze", action="store_true", help="Analyze "
                           "given data and return")
    argparser.add_argument("--model_path", type=str, help="path to model "
                           "directory")
    argparser.add_argument("--prediction_path", type=str, help="path to output "
                           "file of prediction")
    argparser.add_argument("--train", action="store_true", help="train the "
                           "tagger on the given data")
    argparser.add_argument("--feature_template",
                           type=str, default="baseline", help="feature template"
                           " (default: %(default)s)")
    argparser.add_argument("--embedding_path", type=str, help="path to word "
                           "embeddings")
    argparser.add_argument("--bitstring_path", type=str, help="path to word "
                           "bit strings (from a hierarchy of word types)")
    argparser.add_argument("--quiet", action="store_true", help="no messages")
    argparser.add_argument("--dev_path", type=str, help="path to development "
                           "data (used for training)")
    argparser.add_argument("--active", action="store_true", help="perform "
                           "active learning on the given data")
    argparser.add_argument("--active_output_path", type=str, help="path to "
                           "output directory for active learning")
    argparser.add_argument("--active_seed_size",
                           type=int, default=1, help="number of seed examples "
                           "for active learning (default: %(default)d)")
    argparser.add_argument("--active_step_size",
                           type=int, default=1, help="number of examples for "
                           "labeling at each iteration in active learning "
                           "(default: %(default)d)")
    argparser.add_argument("--active_output_interval",
                           type=int, default=100, help="output actively "
                           "selected examples every time this value divides "
                           "their number (default: %(default)d)")
    parsed_args = argparser.parse_args()
    main(parsed_args)
