all: describe test sdist

describe:
	@git describe --always --abbrev=4 2>/dev/null > VERSION
	@echo "release version is: `cat VERSION`"

test:
	@python setup.py test

sdist:
	@python setup.py sdist --format=bztar
