.PHONY: install run test pipeline all help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install   - Install Python dependencies from requirements.txt"
	@echo "  make run       - Launch the interactive Streamlit web dashboard"
	@echo "  make test      - Run the unit test suite with pytest"
	@echo "  make pipeline  - Run the end-to-end ML churn analysis pipeline"
	@echo "  make all       - Install dependencies, run tests, and start Streamlit app"

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/

pipeline:
	python churn_analysis.py

api:
	uvicorn api:app --reload --port 8000

all: install test run
