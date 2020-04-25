test:
	python3 tutucc.py tests > tmp.s
	echo 'int char_fn() { return 257; } int static_fn() { return 5; }' | \
          gcc -xc -c -o tmp2.o -
	gcc -static -o tmp tmp.s tmp2.o
	./tmp