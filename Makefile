all: install

install:
	@pip3 wheel --no-deps .
	@rename 's/py3-none-any/cp310-cp310-manylinux1_x86_64/' *.whl

clean:
	@echo "cleaning..."
	@rm -rf build/ src/xyston.egg-info *.whl
	@find src/ -name '__pycache__' -type d -exec rm -rf {} \;

.PHONY: clean install
