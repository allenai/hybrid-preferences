.PHONY: format

check_dirs := scripts src

format:
	python3 -m black $(check_dirs)
	python3 -m isort $(check_dirs) --profile black

lint:
	python3 -m flake8

create-beaker:
	beaker image delete ljm/human-datamodel
	beaker image create human-datamodel --name human-datamodel --description "From https://github.com/allenai/human-pref-datamodel"