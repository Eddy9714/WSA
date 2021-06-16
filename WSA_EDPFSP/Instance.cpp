#pragma once
#include "Instance.h"

using namespace std;

Instance::Instance(string path) {

	ifstream file(path);

	if (file.is_open()) {
		file >> jobs >> factories >> machines;

		pt = new unsigned short* [jobs];
		s = new unsigned short* [jobs];

		ip = new double[machines];
		sp = new double[machines];

		v = new double[5];
		pV = new double[5];

		for (unsigned short i = 0; i < 5; i++) {
			file >> v[i];
		}

		for (unsigned short i = 0; i < 5; i++) {
			file >> pV[i];
			pV[i] /= 60;
		}

		for (unsigned short i = 0; i < jobs; i++) {
			pt[i] = new unsigned short[machines];
			for (unsigned short m = 0; m < machines; m++) {
				file >> pt[i][m];
			}
		}

		for (unsigned short i = 0; i < jobs; i++) {
			s[i] = new unsigned short[machines];
			for (unsigned short m = 0; m < machines; m++) {
				file >> s[i][m];
			}
		}

		for (unsigned short m = 0; m < machines; m++) {
			file >> ip[m];
			ip[m] /= 60;
		}


		for (unsigned short m = 0; m < machines; m++) {
			file >> sp[m];
			sp[m] /= 60;
		}
	}
	else {
		cerr << "File non found!" << endl << endl;
		exit(-2);
	}
}

Instance::~Instance(){

	for (int i = 0; i < jobs; i++) {
		delete[] pt[i];
		delete[] s[i];
	}

	delete[] pt;
	delete[] s;

	delete[] pV;
	delete[] v;
}