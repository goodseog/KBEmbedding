all: train_TransE test_TransE
train_TransE: train_TransE.cpp
	g++ train_TransE.cpp -o train_TransE -O2 -std=c++17 -Wunused-result

test_TransE: test_TransE.cpp
	g++ test_TransE.cpp -o test_TransE -O2 -std=c++17 -Wunused-result

clean : train_TransE test_TransE
	rm train_TransE test_TransE