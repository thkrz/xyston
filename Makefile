install:
	@pip3 wheel -w build/ --no-deps .

clean:
	@echo "cleaning..."
	@rm -rf build/
	@find src/ -name '*.egg-info' -type d -exec rm -rf {} \;
	@find src/ -name '__pycache__' -type d -exec rm -rf {} \;

.PHONY: clean install
