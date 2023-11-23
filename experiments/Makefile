

# Run the yandex-deep benchmark and log everything to a logs.txt file 
yandex-deep-bench: install-lib 
	poetry run python run-big-bench.py \
		--dataset path/to/yandex-deep.350.fbin \
		--queries path/to/yandex-deep-queries.fbin \
		--gtruth path/to/yandex-ground-truth \
		--metric l2  \
		logs.txt 2>&1


# Install all dependencies including flatnav
install-lib: generate-wheel 
	poetry add ../flatnav_python/dist/*.whl 
	poetry install --no-root

# This will generate the wheel for flatnav and put it in
# ../flatnav_python/dist
generate-wheel:
	pwd 
	cd .. && cd flatnav_python && ./install_flatnav.sh 
	