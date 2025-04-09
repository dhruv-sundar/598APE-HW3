FUNC := clang++
copt := -c 
OBJ_DIR := ./bin/
FLAGS := -O3 -lm -gdwarf-4 -Werror -lstdc++ -fopenmp

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(CPP_FILES:.cpp=.obj)))

all:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS)

clean:
	rm -f ./*.exe

test:
	echo "Running test 1: 1000 planets, 5000 timesteps"
	./main.exe 1000 5000
	echo "Running test 2: 5 planets, 1000000000 timesteps"
	./main.exe 5 1000000000
	echo "Running test 3: 1000 planets, 100000 timesteps"
	./main.exe 1000 100000