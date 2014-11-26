all = liblinear_python

liblinear_python:
	make -C liblinear-1.96 lib

.PHONY: clean
clean:
	make -C liblinear-1.96 clean
	find . -name '*.pyc' -type f -delete
	find . -name '__pycache__' -type d | xargs rm -rf
