#include <cstring>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#define CEIL64(X) ((X + 63) & ~63)

int              nplanets;
int              timesteps;
constexpr double dt = 0.001;
constexpr double G  = 6.6743;

struct PlanetCoords {
    double* x;
    double* y;
    double* vx;
    double* vy;

    PlanetCoords(int num_planets)
        : x((double*)std::aligned_alloc(64UL, CEIL64(sizeof(double) * num_planets))),
          y((double*)std::aligned_alloc(64UL, CEIL64(sizeof(double) * num_planets))),
          vx((double*)std::aligned_alloc(64UL, CEIL64(sizeof(double) * num_planets))),
          vy((double*)std::aligned_alloc(64UL, CEIL64(sizeof(double) * num_planets))) {}

    ~PlanetCoords() {
        free(x);
        free(y);
        free(vx);
        free(vy);
    }

    PlanetCoords& operator=(const PlanetCoords& other) {
        std::memcpy(x, other.x, nplanets * sizeof(double));
        std::memcpy(y, other.y, nplanets * sizeof(double));
        std::memcpy(vx, other.vx, nplanets * sizeof(double));
        std::memcpy(vy, other.vy, nplanets * sizeof(double));
        return *this;
    }
};

float tdiff(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

unsigned long long seed = 100;

unsigned long long randomU64() {
    seed ^= (seed << 21);
    seed ^= (seed >> 35);
    seed ^= (seed << 4);
    return seed;
}

double randomDouble() {
    unsigned long long next = randomU64();
    next >>= (64 - 26);
    unsigned long long next2 = randomU64();
    next2 >>= (64 - 26);
    return ((next << 27) + next2) / (double)(1LL << 53);
}

void next(const PlanetCoords& planets, PlanetCoords& nextplanets, const double* planet_masses) {
    nextplanets                        = planets;
    constexpr auto ELEMS_PER_CACHELINE = 64UL / sizeof(double);
    constexpr auto TILE_SIZE           = ELEMS_PER_CACHELINE * 8;

#pragma omp      parallel for
#pragma omp tile sizes(TILE_SIZE)
    for (int i = 0; i < nplanets; ++i) {
        double accum_vx = 0;
        double accum_vy = 0;
        double planet_x = planets.x[i];
        double planet_y = planets.y[i];
        double planet_mass = planet_masses[i];
#pragma omp simd
        for (int j = 0; j < nplanets; j++) {
            double dx       = planets.x[j] - planet_x;
            double dy       = planets.y[j] - planet_y;
            double distSqr  = dx * dx + dy * dy + 0.0001;
            double sqrt_reciprocal = 1.0 / sqrt(distSqr);
            double invDist  = planet_mass * planet_masses[j] * sqrt_reciprocal;
            double invDist3 = invDist * invDist * invDist;
            accum_vx += dt * dx * invDist3;
            accum_vy += dt * dy * invDist3;
        }
        nextplanets.x[i] += dt * accum_vx;
        nextplanets.y[i] += dt * accum_vy;

        nextplanets.vx[i] += accum_vx;
        nextplanets.vy[i] += accum_vy;
    }
}

int main(int argc, const char** argv) {
    if (argc < 2) {
        printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
        return 1;
    }
    nplanets  = atoi(argv[1]);
    timesteps = atoi(argv[2]);

    PlanetCoords planets(nplanets), nextplanets(nplanets);
    double*      planet_masses = (double*)malloc(sizeof(double) * nplanets);
    for (int i = 0; i < nplanets; i++) {
        planet_masses[i] = randomDouble() * 10 + 0.2;
        planets.x[i]     = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
        planets.y[i]     = (randomDouble() - 0.5) * 100 * pow(1 + nplanets, 0.4);
        planets.vx[i]    = randomDouble() * 5 - 2.5;
        planets.vy[i]    = randomDouble() * 5 - 2.5;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < (timesteps & ~0x1); i++) {
        next(planets, nextplanets, planet_masses);
        next(nextplanets, planets, planet_masses);
        // printf("x=%f y=%f vx=%f vy=%f\n", planets[nplanets-1].x, planets[nplanets-1].y,
        // planets[nplanets-1].vx, planets[nplanets-1].vy);
    }
    if (timesteps & 0x1) {
        next(planets, nextplanets, planet_masses);
    }
    gettimeofday(&end, NULL);
    printf("Total time to run simulation %0.6f seconds, final location %f %f\n",
           tdiff(&start, &end), nextplanets.x[nplanets - 1], nextplanets.y[nplanets - 1]);

    return 0;
}