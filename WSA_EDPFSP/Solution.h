#pragma once
#include <vector>
#include <iostream>

#include "Globals.h"

using namespace std;

class Solution {

	public:
		Solution(unsigned short, unsigned short, unsigned short, unsigned short, bool = false);

		void random();
		void randomPermutation();
		void randomJobAssignment();
		void randomVelocities();

		void print();
		void rawPrint();

		void copy(Solution*);

		unsigned short nJ;
		unsigned short nF;
		unsigned short nM;
		unsigned short mV;

		vector<unsigned short> permutation;
		vector<unsigned short> factoryAssignment;
		vector<vector<unsigned short>> velocities;
};