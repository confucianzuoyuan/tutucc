test:
	python3 tutucc.py tests > tmp.s
	gcc -static -o tmp tmp.s
	./tmp