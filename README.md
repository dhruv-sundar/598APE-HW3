# RUNNING THE CODE INSTRUCTIONS
The paper refers to a final deviated, and a final result preserving version.

The final commit with results preserved is hash `9c4cd0bde2ad9e0280f543bb488cfd68f8afeb18`

The final commit with results deviated is hash `e4d42c2054df40d004f1e3e57fd53d25958a0e36`

To reproduce the results, run the following commands:
```bash
git checkout <hash>
make clean
make
./main.exe 1000 5000
```

The other test cases take pretty annoyingly long to run (1 billion timesteps) so I recommend running with 10k, 100k, 1M timesteps and seeing that it linearly scales, and extrapolate from there.

# 598APE-HW3

This repository contains code for homework 3 of 598APE.

This assignment is relatively simple in comparison to HW1 and HW2 to ensure you have enough time to work on the course project.

In particular, this repository is an implementation of an n-body simulator.

To compile the program run:
```bash
make -j
```

To clean existing build artifacts run:
```bash
make clean
```

This program assumes the following are installed on your machine:
* A working C compiler (g++ is assumed in the Makefile)
* make

The nbody program is a classic physics simulation whose exact results are unable to be solved for exactly through integration.

Here we implement a simple time evolution where each iteration advances the simulation one unit of time, according to Newton's law of gravitation.

Once compiled, one can call the nbody program as follows, where nplanets is the number of randomly generated planets for the simulation, and timesteps denotes how long to run the simulation for:
```bash
./main.exe <nplanets> <timesteps>
```

In particular, consider speeding up simple run like the following (which runs ~6 seconds on my local laptop under the default setup):
```bash
./main.exe 1000 5000
```

Exact bitwise reproducibility is not required, but approximate correctness (within a reasonable region of the final location).