.PHONY: install run test pipeline api frontend all help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install   - Install Python dependencies from requirements.txt"
	@echo "  make run       - Launch the interactive Streamlit web dashboard"
	@echo "  make test      - Run the unit test suite with pytest"
	@echo "  make pipeline  - Run the end-to-end ML churn analysis pipeline"
	@echo "  make api       - Launch the FastAPI prediction backend server"
	@echo "  make frontend  - Launch the Next.js predictor web interface"
	@echo "  make all       - Install dependencies, run tests, and start Streamlit app"

install:
	pip install -r requirements.txt

run:
	streamlit run app/app.py

test:
	pytest tests/

pipeline:
	python src/churn_analysis.py

api:
	uvicorn src.api:app --reload --port 8000

frontend:
	cd frontend && npm run dev

all: install test run
