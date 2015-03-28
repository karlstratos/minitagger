Minitagger (Python 3 + Numpy)
=============================
Minitagger is a tagger for words in sentences. Underlying the tagger is an
implementation of a multi-class SVM (Fan et al., 2008). It makes *independent*
predictions based on *local* context. Even though this method is completely
unstructured (as opposed to CRFs), with the addition of lexical representations
it performs as well as structured models on certain problems like POS tagging.

For experimental details, see: Simple Semi-Supervised POS Tagging (Stratos and
Collins, 2015). You can obtain the word representations used in the experiments
at: http://www.cs.columbia.edu/~stratos/research/wordrep.tar.gz.

Highlights
----------
Minitagger can:

1. Utilize bit string (Brown clusters) and real-valued (word embeddings) lexical
features.
 * These lexical features must include a representation for unknown words.
By default, symbol "<?>" denotes this representation.

2. Train from partially or completely labeled data, of form (an empty line
marks the end of a sentence):

      	 The
      	 dog
      	 saw	V
      	 the
      	 cat

3. Perform *active learning* using whatever features it's equipped with.

Usage
--------
First, type `make` to compile the liblinear package.

### Training and prediction

* Try training a tagger with baseline features:

`python3 minitagger.py example/example.train --model_path /tmp/example.model.baseline --train --feature_template baseline`

* Try training a tagger with bit string features:

`python3 minitagger.py example/example.train --model_path /tmp/example.model.bitstring --train --feature_template bitstring --bitstring_path example/example.bitstring`

* Try training a tagger with embedding features:

`python3 minitagger.py example/example.train --model_path /tmp/example.model.embedding --train --feature_template embedding --embedding_path example/example.embedding`

Then try tagging test data:

`python3 minitagger.py example/example.test --model_path [model] --prediction_path /tmp/example.test.prediction`

### Active learning

* Try active learning with baseline features, seed size 1, and step size 1
(you can also provide a held-out dataset to monitor the improvement in a log
file):

`python3 minitagger.py example/example.train --train --feature_template baseline --active --active_output_path /tmp/active.baseline.seed1.step1 --active_seed_size 1 --active_step_size 1 --active_output_interval 1`

Once you have actively selected examples, you can simply provide these partially
labeled sentences as training data to train a model.