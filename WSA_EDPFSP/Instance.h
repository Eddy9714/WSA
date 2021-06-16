#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "Globals.h"

using namespace std;

class Instance {
	public:

		unsigned short machines;
		unsigned short factories;
		unsigned short jobs;
		unsigned short nVelocities = 5;

		unsigned short** pt; //production time
		unsigned short** s; //setup time

		double* ip; //idle power
		double* sp; //setup power

		double* pV; // power per velocity;
		double* v; // velocities;

		Instance(string path);
		~Instance();
};