#pragma once
#include "Random.h"

extern Random randGen;

template <typename T>
struct Point {
	T x;
	T y;
};

template <typename T>
struct Coordinate {
	T x;
	T y;
	T z;
};

template<typename T1, typename T2>
struct MPoint {
	T1 x;
	T2 y;
};

template<typename T>
unsigned int binarySearch(vector<T>& v, T element) {

	unsigned int l = 0, r = v.size() - 1, m = 0;

	while (l <= r) {

		m = (l + r) / 2;

		if (l == r) {
			if (v[m] <= element) return m + 1;
			else return m;
		}

		if (v[m] <= element) {
			l = m + 1;
		}
		else r = m;
	}

	return 0;
}