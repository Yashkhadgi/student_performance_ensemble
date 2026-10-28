.PHONY: demo data train train-demo experiments experiments-demo tests app verify clean

verify:
	python -m src.data --verify data/raw/student-mat.csv

data:
	python -m src.data --download

demo:
	python -m src.data --demo

train:
	python -m src.train --data data/raw/student-mat.csv

train-demo:
	python -m src.train --demo --fast

experiments:
	python -m src.experiments --data data/raw/student-mat.csv

experiments-demo:
	python -m src.experiments --demo

tests:
	python -m pytest -q

app:
	streamlit run app.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
