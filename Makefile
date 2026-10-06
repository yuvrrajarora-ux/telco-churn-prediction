.PHONY: install data train test api app docker
install:
	pip install -r requirements.txt
data:
	python -m src.generate_data
train:
	python -m src.train
test:
	pytest -q
api:
	uvicorn api.main:app --reload
app:
	streamlit run app/streamlit_app.py
docker:
	docker build -t churn-api . && docker run -p 8000:8000 churn-api
