install:
	@pip3 wheel --no-deps .
	@rename 's/none-any/manylinux1_x86_64/' *.whl

clean:
	@echo "cleaning..."
	@rm -rf build/ src/xyston.egg-info
	@find src/ -name '__pycache__' -type d -exec rm -rf {} \;

.PHONY: clean install
