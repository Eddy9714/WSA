#pragma once
#include <random>

using namespace std;

class Random {
	public:
		mt19937 gen;

		Random() {
			random_device rd;
			gen.seed(rd());
		}

		Random(unsigned int seed) {
			gen.seed(seed);
		}

		int randIntU(int a, int b) {
			uniform_int_distribution<int> distrib(a, b);
			return distrib(gen);
		}

		void twoRandomIndexes(unsigned int d, unsigned int& i1, unsigned int& i2) {
			i1 = randIntU(0, d - 1);
			i2 = (i1 + 1 + randIntU(0, d - 2)) % d;
		}

		double cauchy(double a, double b) {
			std::cauchy_distribution<double> distrib(a, b);
			return distrib(gen);
		}

		double normal(double m, double s) {
			std::normal_distribution<double> distrib(m, s);
			return distrib(gen);
		}

		double randDouble(double a, double b) {
			uniform_real_distribution<double> distrib(a, b);
			return distrib(gen);
		}

		void setSeed(unsigned int seed) {
			gen.seed(seed);
		}
};