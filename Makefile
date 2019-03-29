freeze:
	@pip freeze > requirements.tx

setup:
	@python3 -m venv ./env
	@source ./env/bin/activate
	@pip install -r requirements.txt

install:
	@pip install -r requirements.txt

learn:
	@python oracle.py
