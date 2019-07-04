.PHONY: test build upload distribute

test:
	python setup.py test

build:
	python setup.py sdist bdist_wheel

upload:
	PVER=$(python setup.py --version)
	twine upload dist/latools-$PVER*

distribute:
	make test
	make build
	make upload