import numpy
import tempfile
import unittest
from minitagger import FRONT_BUFFER_SYMBOL
from minitagger import END_BUFFER_SYMBOL
from minitagger import get_baseline_features
from minitagger import get_bitstring_features
from minitagger import get_embedding_features

class TestFeatures(unittest.TestCase):
    """Test the correctness of feature templates."""
    def setUp(self):
        self.word_sequence = ["Tom", "received", "0.3", "?!"]

    def test_get_baseline_features(self):
        """
        Thoroughly test get_baseline_features since this is what other feature
        templates build on.
        """
        Tom_features = get_baseline_features(self.word_sequence, 0)
        self.assertEqual(len(Tom_features), 16)
        self.assertEqual(Tom_features["word(0)=Tom"], 1)
        self.assertEqual(Tom_features["is_capitalized(0)=True"], 1)
        self.assertEqual(Tom_features["prefix1(0)=T"], 1)
        self.assertEqual(Tom_features["prefix2(0)=To"], 1)
        self.assertEqual(Tom_features["prefix3(0)=Tom"], 1)
        self.assertEqual(Tom_features["prefix4(0)=Tom*"], 1)
        self.assertEqual(Tom_features["suffix1(0)=m"], 1)
        self.assertEqual(Tom_features["suffix2(0)=om"], 1)
        self.assertEqual(Tom_features["suffix3(0)=Tom"], 1)
        self.assertEqual(Tom_features["suffix4(0)=*Tom"], 1)
        self.assertEqual(Tom_features["is_all_nonalphanumeric(0)=False"], 1)
        self.assertEqual(Tom_features["is_float(0)=False"], 1)
        self.assertEqual(Tom_features["word(-1)=" + FRONT_BUFFER_SYMBOL], 1)
        self.assertEqual(Tom_features["word(-2)=" + FRONT_BUFFER_SYMBOL], 1)
        self.assertEqual(Tom_features["word(+1)=received"], 1)
        self.assertEqual(Tom_features["word(+2)=0.3"], 1)

        received_features = get_baseline_features(self.word_sequence, 1)
        self.assertEqual(len(received_features), 16)
        self.assertEqual(received_features["word(0)=received"], 1)
        self.assertEqual(received_features["is_capitalized(0)=False"], 1)
        self.assertEqual(received_features["prefix1(0)=r"], 1)
        self.assertEqual(received_features["prefix2(0)=re"], 1)
        self.assertEqual(received_features["prefix3(0)=rec"], 1)
        self.assertEqual(received_features["prefix4(0)=rece"], 1)
        self.assertEqual(received_features["suffix1(0)=d"], 1)
        self.assertEqual(received_features["suffix2(0)=ed"], 1)
        self.assertEqual(received_features["suffix3(0)=ved"], 1)
        self.assertEqual(received_features["suffix4(0)=ived"], 1)
        self.assertEqual(received_features["is_all_nonalphanumeric(0)=False"],
                         1)
        self.assertEqual(received_features["is_float(0)=False"], 1)
        self.assertEqual(received_features["word(-1)=Tom"], 1)
        self.assertEqual(received_features["word(-2)=" + FRONT_BUFFER_SYMBOL],
                         1)
        self.assertEqual(received_features["word(+1)=0.3"], 1)
        self.assertEqual(received_features["word(+2)=?!"], 1)

        float_features = get_baseline_features(self.word_sequence, 2)
        self.assertEqual(len(float_features), 16)
        self.assertEqual(float_features["word(0)=0.3"], 1)
        self.assertEqual(float_features["is_capitalized(0)=False"], 1)
        self.assertEqual(float_features["prefix1(0)=0"], 1)
        self.assertEqual(float_features["prefix2(0)=0."], 1)
        self.assertEqual(float_features["prefix3(0)=0.3"], 1)
        self.assertEqual(float_features["prefix4(0)=0.3*"], 1)
        self.assertEqual(float_features["suffix1(0)=3"], 1)
        self.assertEqual(float_features["suffix2(0)=.3"], 1)
        self.assertEqual(float_features["suffix3(0)=0.3"], 1)
        self.assertEqual(float_features["suffix4(0)=*0.3"], 1)
        self.assertEqual(float_features["is_all_nonalphanumeric(0)=False"], 1)
        self.assertEqual(float_features["is_float(0)=True"], 1)
        self.assertEqual(float_features["word(-1)=received"], 1)
        self.assertEqual(float_features["word(-2)=Tom"], 1)
        self.assertEqual(float_features["word(+1)=?!"], 1)
        self.assertEqual(float_features["word(+2)=" + END_BUFFER_SYMBOL], 1)

        symbol_features = get_baseline_features(self.word_sequence, 3)
        self.assertEqual(len(symbol_features), 16)
        self.assertEqual(symbol_features["word(0)=?!"], 1)
        self.assertEqual(symbol_features["is_capitalized(0)=False"], 1)
        self.assertEqual(symbol_features["prefix1(0)=?"], 1)
        self.assertEqual(symbol_features["prefix2(0)=?!"], 1)
        self.assertEqual(symbol_features["prefix3(0)=?!*"], 1)
        self.assertEqual(symbol_features["prefix4(0)=?!**"], 1)
        self.assertEqual(symbol_features["suffix1(0)=!"], 1)
        self.assertEqual(symbol_features["suffix2(0)=?!"], 1)
        self.assertEqual(symbol_features["suffix3(0)=*?!"], 1)
        self.assertEqual(symbol_features["suffix4(0)=**?!"], 1)
        self.assertEqual(symbol_features["is_all_nonalphanumeric(0)=True"], 1)
        self.assertEqual(symbol_features["is_float(0)=False"], 1)
        self.assertEqual(symbol_features["word(-1)=0.3"], 1)
        self.assertEqual(symbol_features["word(-2)=received"], 1)
        self.assertEqual(symbol_features["word(+1)=" + END_BUFFER_SYMBOL], 1)
        self.assertEqual(symbol_features["word(+2)=" + END_BUFFER_SYMBOL], 1)

    def test_get_bitstring_features(self):
        """
        Only check features about bit strings.
        """
        bitstring_dictionary = {"received": "1",
                                "0.3": "00",
                                "Tom": "010",
                                "<?>": "011"}
        Tom_features = get_bitstring_features(self.word_sequence, 0,
                                              bitstring_dictionary)
        self.assertEqual(Tom_features["bitstring(0)_prefix(1)=0"], 1)
        self.assertEqual(Tom_features["bitstring(0)_prefix(2)=01"], 1)
        self.assertEqual(Tom_features["bitstring(0)_prefix(3)=010"], 1)
        self.assertEqual(Tom_features["bitstring(0)_all=010"], 1)
        self.assertEqual(Tom_features["bitstring(+1)_prefix(1)=1"], 1)
        self.assertEqual(Tom_features["bitstring(+1)_all=1"], 1)

        received_features = get_bitstring_features(self.word_sequence, 1,
                                                   bitstring_dictionary)
        self.assertEqual(received_features["bitstring(-1)_prefix(1)=0"], 1)
        self.assertEqual(received_features["bitstring(-1)_prefix(2)=01"], 1)
        self.assertEqual(received_features["bitstring(-1)_prefix(3)=010"], 1)
        self.assertEqual(received_features["bitstring(-1)_all=010"], 1)
        self.assertEqual(received_features["bitstring(0)_prefix(1)=1"], 1)
        self.assertEqual(received_features["bitstring(0)_all=1"], 1)

        float_features = get_bitstring_features(self.word_sequence, 2,
                                                bitstring_dictionary)
        self.assertEqual(float_features["bitstring(0)_prefix(1)=0"], 1)
        self.assertEqual(float_features["bitstring(0)_prefix(2)=00"], 1)
        self.assertEqual(float_features["bitstring(0)_all=00"], 1)
        self.assertEqual(float_features["bitstring(-1)_prefix(1)=1"], 1)
        self.assertEqual(float_features["bitstring(-1)_all=1"], 1)
        self.assertEqual(float_features["bitstring(+1)_prefix(1)=0"], 1)
        self.assertEqual(float_features["bitstring(+1)_prefix(2)=01"], 1)
        self.assertEqual(float_features["bitstring(+1)_prefix(3)=011"], 1)
        self.assertEqual(float_features["bitstring(+1)_all=011"], 1)

        symbol_features = get_bitstring_features(self.word_sequence, 3,
                                                 bitstring_dictionary)
        self.assertEqual(symbol_features["bitstring(-1)_prefix(1)=0"], 1)
        self.assertEqual(symbol_features["bitstring(-1)_prefix(2)=00"], 1)
        self.assertEqual(symbol_features["bitstring(-1)_all=00"], 1)
        self.assertEqual(symbol_features["bitstring(0)_prefix(1)=0"], 1)
        self.assertEqual(symbol_features["bitstring(0)_prefix(2)=01"], 1)
        self.assertEqual(symbol_features["bitstring(0)_prefix(3)=011"], 1)
        self.assertEqual(symbol_features["bitstring(0)_all=011"], 1)

    def test_get_embedding_features(self):
        """
        Only check features about embeddings.
        """
        embedding_dictionary = {}
        embedding_dictionary["received"] = numpy.array([-1, -1])
        embedding_dictionary["0.3"] = numpy.array([0.1, 0.3])
        embedding_dictionary["Tom"] = numpy.array([0.8, 0.9])
        embedding_dictionary["<?>"] = numpy.array([1, 1])

        Tom_features = get_embedding_features(self.word_sequence, 0,
                                              embedding_dictionary)
        self.assertEqual(Tom_features["embedding(0)_at(1)"], 0.8)
        self.assertEqual(Tom_features["embedding(0)_at(2)"], 0.9)
        self.assertEqual(Tom_features["embedding(+1)_at(1)"], -1)
        self.assertEqual(Tom_features["embedding(+1)_at(2)"], -1)

        received_features = get_embedding_features(self.word_sequence, 1,
                                                   embedding_dictionary)
        self.assertEqual(received_features["embedding(-1)_at(1)"], 0.8)
        self.assertEqual(received_features["embedding(-1)_at(2)"], 0.9)
        self.assertEqual(received_features["embedding(0)_at(1)"], -1)
        self.assertEqual(received_features["embedding(0)_at(2)"], -1)
        self.assertEqual(received_features["embedding(+1)_at(1)"], 0.1)
        self.assertEqual(received_features["embedding(+1)_at(2)"], 0.3)

        float_features = get_embedding_features(self.word_sequence, 2,
                                                embedding_dictionary)
        self.assertEqual(float_features["embedding(-1)_at(1)"], -1)
        self.assertEqual(float_features["embedding(-1)_at(2)"], -1)
        self.assertEqual(float_features["embedding(0)_at(1)"], 0.1)
        self.assertEqual(float_features["embedding(0)_at(2)"], 0.3)
        self.assertEqual(float_features["embedding(+1)_at(1)"], 1)
        self.assertEqual(float_features["embedding(+1)_at(2)"], 1)

        symbol_features = get_embedding_features(self.word_sequence, 3,
                                                 embedding_dictionary)
        self.assertEqual(symbol_features["embedding(-1)_at(1)"], 0.1)
        self.assertEqual(symbol_features["embedding(-1)_at(2)"], 0.3)
        self.assertEqual(symbol_features["embedding(0)_at(1)"], 1)
        self.assertEqual(symbol_features["embedding(0)_at(2)"], 1)

if __name__ == '__main__':
    unittest.main()
