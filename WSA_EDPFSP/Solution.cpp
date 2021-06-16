#include "Solution.h"

Solution::Solution(unsigned short nJobs, unsigned short nFactories, unsigned short nMachines, unsigned short maxVelocity, bool init) :
 velocities(nMachines) {
	nJ = nJobs;
	nF = nFactories;
	nM = nMachines;
	mV = maxVelocity;

	if (init) {
		permutation = vector<unsigned short>(nJ, 0);
		factoryAssignment = vector<unsigned short>(nJ, 0);
		
		for (unsigned short i = 0; i < velocities.size(); i++) {
			velocities[i] = vector<unsigned short>(nJ, 0);
		}
	}
	else {
		permutation.reserve(nJ);
		factoryAssignment.reserve(nJ);

		for (unsigned short i = 0; i < velocities.size(); i++) {
			velocities[i].reserve(nJ);
		}
	}
	
}

void Solution::random() {
	randomPermutation();
	randomJobAssignment();
	randomVelocities();
}

void Solution::randomPermutation() {

	if (permutation.size() == 0) {
		for (unsigned short i = 0; i < nJ; i++) {
			permutation.push_back(i);
		}
	}

	unsigned short tmp, randomValue;

	for (unsigned short k = permutation.size() - 1; k > 0; k--) {
		randomValue = randGen.randIntU(0, k);
		tmp = permutation[randomValue];
		permutation[randomValue] = permutation[k];
		permutation[k] = tmp;
	}
}

void Solution::randomJobAssignment() {

	if (factoryAssignment.size() == 0) {
		for (unsigned short i = 0; i < nJ; i++) {
			factoryAssignment.push_back(randGen.randIntU(0, nF - 1));
		}
	}
	else {
		for (unsigned short i = 0; i < nJ; i++) {
			factoryAssignment[i] = randGen.randIntU(0, nF - 1);
		}
	}
}

void Solution::randomVelocities() {

	unsigned short j;

	for (unsigned short i = 0; i < velocities.size(); i++) {

		if (velocities[i].size() == 0) {
			for (j = 0; j < nJ; j++) {
				velocities[i].push_back(randGen.randIntU(0, mV - 1));
			}
		}
		else {
			for (j = 0; j < nJ; j++) {
				velocities[i][j] = randGen.randIntU(0, mV - 1);
			}
		}		
	}
}

void Solution::copy(Solution* s) {
	
	permutation = s->permutation;
	factoryAssignment = s->factoryAssignment;
	velocities = s->velocities;

	nJ = s->nJ;
	nF = s->nF;
	nM = s->nM;
	mV = s->mV;
}

void Solution::rawPrint() {

	cout << "[";
	for (unsigned short i = 0; i < permutation.size(); i++) {
		cout << permutation[i] << ",";
	}
	cout << "\b]" << endl;

	cout << "[";
	for (unsigned short i = 0; i < factoryAssignment.size(); i++) {
		cout << factoryAssignment[i] << ",";
	}
	cout << "\b]" << endl << endl;

	unsigned short j;

	for (unsigned short i = 0; i < velocities.size(); i++) {

		for (j = 0; j < velocities[i].size(); j++) {
			cout << velocities[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

}

void Solution::print() {
	
	vector<vector<unsigned short>> factories(nF);

	for (unsigned short i = 0; i < permutation.size(); i++) {
		factories[factoryAssignment[permutation[i]]].push_back(permutation[i]);
	}
	
	cout << "[";

	for (unsigned short f = 0; f < factories.size(); f++) {
		cout << "{";
		for (unsigned short i = 0; i < factories[f].size(); i++) {
			cout << factories[f][i] << ",";
		}
		cout << "\b}";
	}

	cout << "]" << endl << endl;

	unsigned short j;

	for (unsigned short i = 0; i < velocities.size(); i++) {

		for (j = 0; j < velocities[i].size(); j++) {
			cout << velocities[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}