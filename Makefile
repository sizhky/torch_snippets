.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard nbs/*.ipynb)

all: torch_snippets docs

torch_snippets: $(SRC)
	nbdev_build_lib
	touch torch_snippets

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_conda_package
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist

profile-time:
	kernprof -l torch_snippets/__module_timing__.py
	mkdir -p debug
	python -m line_profiler -tm "__module_timing__.py.lprof" | tee debug/profile_time.txt