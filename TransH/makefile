all: train_TransH test_TransH
train_TransH: train_TransH.cpp
	g++ train_TransH.cpp -o train_TransH -O2 -std=c++17 -Wunused-result

test_TransH: test_TransH.cpp
	g++ test_TransH.cpp -o test_TransH -O2 -std=c++17 -Wunused-result

clean : train_TransH test_TransH
	rm train_TransH test_TransH