#include "WSA_EDPFSP.h"

WSA_EDPFSP::WSA_EDPFSP(string path) : instance(Instance(path)) {
	info.positions = vector<vector<unsigned short>>(instance.factories);
	info.scores = vector<Point<double>>(instance.factories);

	c = vector<double>(instance.machines);
	o = vector <vector<Point<double>>>(instance.jobs);

	lsFunctions.push_back([this](Solution* s) -> void {ICFJIS(s); });
	lsFunctions.push_back([this](Solution* s) -> void {ICFJS(s); });
	lsFunctions.push_back([this](Solution* s) -> void {ICFJI(s); });
	lsFunctions.push_back([this](Solution* s) -> void {ECFFS(s); });
	lsFunctions.push_back([this](Solution* s) -> void {ECFJIS(s); });

	for (unsigned short i = 0; i < instance.jobs; i++) {
		o[i] = vector<Point<double>>(instance.machines);
	}
}

vector<MPoint<Solution*, Point<double>>> WSA_EDPFSP::run(unsigned short populationSize, unsigned int nFunctionEvaluations, double pCR, double pM, unsigned int seed) {

	unsigned int nfes = nFunctionEvaluations;
	unsigned int currentNfes = 0;

	if (seed > 0) randGen.setSeed(seed);

	vector<MPoint<Solution*, Point<double>>> population;
	population.reserve(populationSize + 2);

	initPopulation(population, populationSize, currentNfes);

	unsigned int ri1, ri2;
	int bnIndex;

	while (currentNfes < nfes) {

		randGen.twoRandomIndexes(populationSize, ri1, ri2);

		bnIndex = betterAndNearest(population, ri2);

		if (bnIndex != -1) {
			ri2 = bnIndex;
		}

		crossoverAndMutate(population[ri1].x, population[ri2].x, 
			{population[populationSize].x, population[populationSize + 1].x}, pCR, pM);

		if (randGen.randDouble(0, 1) < 0.5) {
			localSearch(population[populationSize], currentNfes);
			localSearch(population[populationSize + 1], currentNfes, false);
		}
		else {
			localSearch(population[populationSize], currentNfes, false);
			localSearch(population[populationSize + 1], currentNfes);
		}

		//ndsa
		ndsa(population);
	}

	delete population[populationSize].x;
	delete population[populationSize + 1].x;
	population.resize(population.size() - 2);

	return population;
}

void WSA_EDPFSP::initPopulation(vector<MPoint<Solution*, Point<double>>>& population, unsigned short nSolutions,
	unsigned int& currentNfes) {

	Solution* dnehSolution = new Solution(instance.jobs, instance.factories, 
		instance.machines, instance.nVelocities, true);

	DNEH(dnehSolution);

	unsigned short dnehCopies = round(nSolutions / 5);

	for (unsigned short i = 0; i < dnehCopies; i++) {
		Solution* s = new Solution(instance.jobs, instance.factories, instance.machines, instance.nVelocities);

		s->permutation = dnehSolution->permutation;
		randomMethod(s);
		s->randomVelocities();

		population.push_back({s, evalSolution(s, currentNfes)->score});
	}

	for (unsigned short i = dnehCopies; i < nSolutions; i++) {
		Solution* s = new Solution(instance.jobs, instance.factories, instance.machines, instance.nVelocities);
		s->randomPermutation();
		randomMethod(s);
		s->randomVelocities();
		population.push_back({s, evalSolution(s, currentNfes)->score });
	}

	//Space for children
	for (unsigned short i = 0; i < 2; i++) {
		Solution* s = new Solution(instance.jobs, instance.factories, instance.machines, instance.nVelocities);
		population.push_back({s});
	}

	delete dnehSolution;
}

void WSA_EDPFSP::DNEH(Solution* s) {

	vector<double> p(instance.jobs);

	for (unsigned short i = 0; i < instance.jobs; i++) {
		p[i] = 0;
		s->permutation[i] = i;
		for (unsigned short j = 0; j < instance.machines; j++) {
			p[i] += instance.pt[i][j];
		}
	}

	sort(s->permutation.begin(), s->permutation.end(), [p](unsigned short i, unsigned short j) {
		return p[i] > p[j];
	});

	vector<vector<unsigned short>> factories(instance.factories);

	for (unsigned short i = 0; i < instance.factories; i++) {
		s->factoryAssignment[s->permutation[i]] = i;
		factories[i].push_back(i);
	}

	MPoint<unsigned short, double> eval;
	MPoint<unsigned short, double> bestEval;
	unsigned short bestFactory;

	for (unsigned short i = instance.factories; i < s->permutation.size(); i++) {
		bestEval.y = DBL_MAX;

		for (unsigned short j = 0; j < instance.factories; j++) {
			eval = bestInsertion(s, factories[j], i);

			if (eval.y < bestEval.y) {
				bestEval = eval;
				bestFactory = j;
			}
		}

		factories[bestFactory].insert(factories[bestFactory].begin() + bestEval.x, i);

		s->factoryAssignment[s->permutation[i]] = bestFactory;
	}

	vector<unsigned short> copy(s->permutation);

	unsigned short counter = 0;
	for (unsigned short i = 0; i < factories.size(); i++) {
		for (unsigned short j = 0; j < factories[i].size(); j++) {
			s->permutation[counter++] = copy[factories[i][j]];
		}
	}
}

void WSA_EDPFSP::randomMethod(Solution* solution) {

	solution->factoryAssignment.resize(instance.jobs);

	for (unsigned short i = 0; i < instance.factories; i++) {
		solution->factoryAssignment[solution->permutation[i]] = i;
	}

	for (unsigned short i = instance.factories; i < instance.jobs; i++) {
		solution->factoryAssignment[solution->permutation[i]] = randGen.randIntU(0, instance.factories - 1);
	}
}


int WSA_EDPFSP::betterAndNearest(vector<MPoint<Solution*, Point<double>>>& population, unsigned short whaleIndex) {
	int chosenIndex = -1;
	unsigned int minDist = UINT32_MAX, dist;

	for (unsigned short i = 0; i < population.size() - 2; i++) {
		if (domination(population[i].y, population[whaleIndex].y) && 
			(dist = distance(population[i].x, population[whaleIndex].x)) < minDist) {
			chosenIndex = i;
			minDist = dist;
		}
	}

	return chosenIndex;
}

bool WSA_EDPFSP::domination(Point<double>& s1, Point<double>& s2) {

	if (s1.x <= s2.x && s1.y <= s2.y && (s1.x < s2.x || s1.y < s2.y))
		return true;
	else return false;
}


unsigned int WSA_EDPFSP::distance(Solution* s1, Solution* s2) {
	unsigned int dis = 0;

	for (unsigned short i = 0; i < s1->permutation.size(); i++) {
		if (s1->permutation[i] != s2->permutation[i])
			dis++;

		if (s1->factoryAssignment[i] != s2->factoryAssignment[i])
			dis++;
	}

	return dis;
}


void WSA_EDPFSP::crossoverAndMutate(Solution* s1, Solution* s2, Point<Solution*> ris, double pCR, double pM) {
	
	ris.x->copy(s1);
	ris.y->copy(s2);

	crossover(s1, s2, ris, pCR);
	
	mutation(ris.x, pM);
	mutation(ris.y, pM);

	normalize(ris.x);
	normalize(ris.y);
}

void WSA_EDPFSP::crossover(Solution* s1, Solution* s2, Point<Solution*>& ris, double pCR) {

	unsigned int i1, i2;
	randGen.twoRandomIndexes(s1->permutation.size(), i1, i2);
	unsigned short imin = min(i1, i2);
	unsigned short imax = max(i1, i2);

	vector<unsigned short> mappingP1(s1->permutation.size());
	vector<unsigned short> mappingP2(s2->permutation.size());

	for (unsigned short i = 0; i < s1->permutation.size(); i++) {
		mappingP1[s1->permutation[i]] = i;
		mappingP2[s2->permutation[i]] = i;
	}

	MPoint<Point<vector<unsigned short>>, Point<unsigned short>> info1 = { {mappingP1, mappingP2}, {imin, imax} };
	MPoint<Point<vector<unsigned short>>, Point<unsigned short>> info2 = { {mappingP2, mappingP1}, {imin, imax} };

	if(randGen.randDouble(0,1) < pCR) pmx(s1->permutation, s2->permutation, ris.x->permutation, info1);
	if (randGen.randDouble(0, 1) < pCR) pmx(s2->permutation, s1->permutation, ris.y->permutation, info2);

	if (randGen.randDouble(0, 1) < pCR) 
		tpx(s1->factoryAssignment, s2->factoryAssignment, { ris.x->factoryAssignment, ris.y->factoryAssignment });

	if (randGen.randDouble(0, 1) < pCR) {
		for (unsigned short i = 0; i < s1->velocities.size(); i++) {
			tpx(s1->velocities[i], s2->velocities[i], { ris.x->velocities[i], ris.y->velocities[i] });
		}
	}
}

void WSA_EDPFSP::pmx(vector<unsigned short>& p1, vector<unsigned short>& p2, vector<unsigned short>& ris, 
	MPoint<Point<vector<unsigned short>>, Point<unsigned short>>& info) {

	unsigned short imin = info.y.x;
	unsigned short imax = info.y.y;

	vector<unsigned short>& mappingP1 = info.x.x;
	vector<unsigned short>& mappingP2 = info.x.y;

	vector<bool> settedPositions(ris.size(), false);
	fill(settedPositions.begin() + imin, settedPositions.begin() + (imax + 1), true);

	unsigned short pos, job, cursor; 

	for (unsigned short i = imin; i <= imax; i++) {
		pos = mappingP1[p2[i]];

		if (pos < imin || pos > imax) {

			cursor = i;
			do {
				job = p1[cursor];
				cursor = mappingP2[job];

			} while (cursor >= imin && cursor <= imax);

			ris[cursor] = p2[i];
			settedPositions[cursor] = true;
		}
	}

	for (unsigned short i = 0; i < settedPositions.size(); i++) {
		if (!settedPositions[i]) {
			ris[i] = p2[i];
		}
	}
}

void WSA_EDPFSP::tpx(vector<unsigned short>& v1, vector<unsigned short>& v2, Point<vector<unsigned short>&> ris) {
	
	unsigned int i1, i2;
	randGen.twoRandomIndexes(v1.size(), i1, i2);
	unsigned short imin = min(i1, i2);
	unsigned short imax = max(i1, i2);

	copy(v2.begin() + imin, v2.begin() + imax + 1, ris.x.begin() + imin);
	copy(v1.begin() + imin, v1.begin() + imax + 1, ris.y.begin() + imin);
}

void WSA_EDPFSP::mutation(Solution* s, double pM) {
	
	unsigned int i1, i2;
	
	if (randGen.randDouble(0, 1) < pM) {
		//permutation mutation
		randGen.twoRandomIndexes(s->permutation.size(), i1, i2);
		unsigned short tmp = s->permutation[i1];
		s->permutation[i1] = s->permutation[i2];
		s->permutation[i2] = tmp;
	}

	if (randGen.randDouble(0, 1) < pM) {
		//assignment mutation
		unsigned short job = s->permutation[randGen.randIntU(0, s->permutation.size() - 1)];
		unsigned short factory = s->factoryAssignment[job];

		unsigned short newFactory = randGen.randIntU(0, s->nF - 2);
		if (newFactory >= factory)
			newFactory++;

		s->factoryAssignment[job] = newFactory;
	}


	if (randGen.randDouble(0, 1) < pM) {
		//speed mutation
		unsigned short job = randGen.randIntU(0, s->permutation.size() - 1);
		unsigned short machine = randGen.randIntU(0, s->nM - 1);
		unsigned short velocity = s->velocities[machine][job];

		unsigned short newVelocity = randGen.randIntU(0, instance.nVelocities - 2);
		if (newVelocity >= velocity)
			newVelocity++;

		s->velocities[machine][job] = newVelocity;
	}
}

void WSA_EDPFSP::normalize(Solution* s) {

	//assign job to empty factories
	vector<vector<unsigned short>> fj(s->nF);
	vector<unsigned short> emptyFactories;
	vector<unsigned short> moreThanOneFactories;

	for (unsigned short i = 0; i < instance.factories; i++) {
		fj[i] = vector<unsigned short>();
	}

	for (unsigned short i = 0; i < s->permutation.size(); i++) {
		fj[s->factoryAssignment[s->permutation[i]]].push_back(s->permutation[i]);
	}

	for (unsigned short i = 0; i < fj.size(); i++) {
		if (fj[i].size() == 0)
			emptyFactories.push_back(i);
		else if (fj[i].size() > 1)
			moreThanOneFactories.push_back(i);
	}

	unsigned short randomMtofPos;
	unsigned short randomMtof;
	unsigned short randomJobPos;
	unsigned short randomJob;

	for (unsigned short i = 0; i < emptyFactories.size(); i++) {
		randomMtofPos = randGen.randIntU(0, moreThanOneFactories.size() - 1);
		randomMtof = moreThanOneFactories[randomMtofPos];
		randomJobPos = randGen.randIntU(0, fj[randomMtof].size() - 1);
		randomJob = fj[randomMtof][randomJobPos];

		fj[randomMtof].erase(fj[randomMtof].begin() + randomJobPos);

		if (fj[randomMtof].size() == 1)
			moreThanOneFactories.erase(moreThanOneFactories.begin() + randomMtofPos);

		s->factoryAssignment[randomJob] = emptyFactories[i];
	}
}

WSA_EDPFSP::InfoFactories* WSA_EDPFSP::evalSolution(Solution* solution, unsigned int& currentNfes, bool countFe) {

	if (countFe) currentNfes++;

	for (unsigned short i = 0; i < instance.factories; i++) {
		info.positions[i] = vector<unsigned short>();
	}

	for (unsigned short i = 0; i < solution->permutation.size(); i++) {
		info.positions[solution->factoryAssignment[solution->permutation[i]]].push_back(i);
	}

	Point<double> output;

	info.score = { 0. , 0. };

	for (unsigned short f = 0; f < info.positions.size(); f++) {

		output = evalPartialSolution(solution, info.positions[f]);
		info.scores[f] = output;

		if (output.x > info.score.x) {
			info.worstFactoryIndex = f;
			info.score.x = output.x;
		}

		info.score.y += output.y;
	}

	return &info;
}

Point<double> WSA_EDPFSP::evalPartialSolution(Solution* solution, vector<unsigned short>& factory) {
	
	if (factory.size() == 0)
		return { 0., 0. };
	 
	unsigned short job;
	double realExecutionTime, energy = 0.;

	for (unsigned short i = 0; i < factory.size(); i++) {

		job = solution->permutation[factory[i]];

		for (unsigned short j = 0; j < instance.machines; j++) {

			realExecutionTime = instance.pt[job][j] / instance.v[solution->velocities[j][job]];
			energy += realExecutionTime * instance.pV[solution->velocities[j][job]];

			realExecutionTime += instance.s[job][j];
			energy += instance.sp[j] * instance.s[job][j];

			if (i == 0 && j == 0) {
				c[j] = realExecutionTime;
			}
			else if (i == 0 && j != 0) {
				c[j] = realExecutionTime + c[j - 1];
			}
			else if (j != 0) {

				if (c[j - 1] > c[j])
					energy += instance.ip[j] * (c[j - 1] - c[j]);

				c[j] = realExecutionTime + max(c[j - 1], c[j]);
			}
			else {
				c[0] = realExecutionTime + c[0];
			}
		}
	}

	double makeSpan = c[instance.machines - 1];

	return { makeSpan , energy };
}

MPoint<unsigned short, double> WSA_EDPFSP::bestInsertion(Solution* solution, vector<unsigned short>& factory, unsigned short posToInsert) {
	
	if (factory.size() == 0) {
		factory.push_back(posToInsert);

		Point<double> evaluation = evalPartialSolution(solution, factory);
		factory.erase(factory.begin());

		return {0, evaluation.x};
	}

	unsigned short jobToInsert = solution->permutation[posToInsert];

	double* e = new double[instance.machines];
	double* f = new double[instance.machines];
	double** q = new double* [factory.size()];

	double bestMakespan = DBL_MAX;
	unsigned short bestPosition = 0;

	unsigned short job;
	double realExecutionTime, realExecutionTime2;

	for (int i = factory.size() - 1; i >= 0; i--) {
		q[i] = new double[instance.machines];

		job = solution->permutation[factory[i]];

		for (int j = instance.machines - 1; j >= 0; j--) {

			realExecutionTime = instance.pt[job][j] / instance.v[solution->velocities[j][job]];
			realExecutionTime += instance.s[job][j];

			if (i != factory.size() - 1 && j != instance.machines - 1) {
				q[i][j] = max(q[i][j + 1], q[i + 1][j]) + realExecutionTime;
			}
			else if (i != factory.size() - 1 && j == instance.machines - 1) {
				q[i][j] = q[i + 1][j] + realExecutionTime;
			}
			else if (j != instance.machines - 1) {
				q[i][j] = q[i][j + 1] + realExecutionTime;
			}
			else {
				q[i][j] = realExecutionTime;
			}
		}
	}

	for (unsigned short i = 0; i <= factory.size(); i++) {

		if(i!=factory.size())
			job = solution->permutation[factory[i]];

		for (unsigned short j = 0; j < instance.machines; j++) {

			if (i != factory.size()) {
				realExecutionTime = instance.pt[job][j] / instance.v[solution->velocities[j][job]];
				realExecutionTime += instance.s[job][j];
			}

			realExecutionTime2 = instance.pt[jobToInsert][j] / instance.v[solution->velocities[j][jobToInsert]];
			realExecutionTime2 += instance.s[jobToInsert][j];

			if (i == 0 && j == 0) {
				f[0] = realExecutionTime2;
				e[0] = realExecutionTime;
			}
			else if (i == 0 && j != 0) {
				f[j] = f[j - 1] + realExecutionTime2;
				e[j] = e[j - 1] + realExecutionTime;
			}
			else if (j == 0) {
				f[0] = e[0] + realExecutionTime2;

				if (i != factory.size())
					e[0] = e[0] + realExecutionTime;
			}
			else {
				f[j] = max(f[j - 1], e[j]) + realExecutionTime2;
				if (i != factory.size())
					e[j] = max(e[j - 1], e[j]) + realExecutionTime;
			}
		}

		double partialMakespan = 0;

		for (unsigned short j = 0; j < instance.machines; j++) {

			if (i != factory.size())
				partialMakespan = max(partialMakespan, f[j] + q[i][j]);
			else
				partialMakespan = max(partialMakespan, f[j]);
		}

		if (partialMakespan < bestMakespan) {
			bestMakespan = partialMakespan;
			bestPosition = i;
		}

	}

	delete[] e;
	delete[] f;

	for (unsigned short i = 0; i < factory.size(); i++) {
		delete[] q[i];
	}

	delete[] q;

	return { bestPosition, bestMakespan };
}

void WSA_EDPFSP::ndsa(vector<MPoint<Solution*, Point<double>>>& population) {

	vector<Individual> ndsaPopulation;
	ndsaPopulation.reserve(population.size());

	for (unsigned short p = 0; p < population.size(); p++) {
		ndsaPopulation.push_back({ p, population[p].x, population[p].y});
	}
	
	//create frontiers based on dominatingCounter
	Point<double> minMaxObj1 = { DBL_MAX, DBL_MIN };
	Point<double> minMaxObj2 = { DBL_MAX, DBL_MIN };

	vector<vector<Individual*>> frontiers(1);

	for (unsigned short p = 0; p < population.size(); p++) {

		if (ndsaPopulation[p].score.x < minMaxObj1.x)
			minMaxObj1.x = ndsaPopulation[p].score.x;

		if (ndsaPopulation[p].score.x > minMaxObj1.y)
			minMaxObj1.y = ndsaPopulation[p].score.x;

		if (ndsaPopulation[p].score.y < minMaxObj2.x)
			minMaxObj2.x = ndsaPopulation[p].score.y;

		if (ndsaPopulation[p].score.y < minMaxObj2.y)
			minMaxObj2.y = ndsaPopulation[p].score.y;

		for (unsigned short q = 0; q < population.size(); q++) {
			if (p != q) {
				if (domination(ndsaPopulation[p].score, ndsaPopulation[q].score)) {
					ndsaPopulation[p].dominated.push_back(&ndsaPopulation[q]);
				}
				else if (domination(ndsaPopulation[q].score, ndsaPopulation[p].score)) {
					ndsaPopulation[p].dominatingCounter++;
				}
			}
		}

		if (ndsaPopulation[p].dominatingCounter == 0)
			frontiers[0].push_back(&ndsaPopulation[p]);
	}

	unsigned short ind = 0, counter;

	while (frontiers.size() > ind) {
		for (unsigned short i = 0; i < frontiers[ind].size(); i++) {
			for (unsigned short j = 0; j < frontiers[ind][i]->dominated.size(); j++) {

				counter = --frontiers[ind][i]->dominated[j]->dominatingCounter;
				
				if (counter == 0) {
					if (frontiers.size() == ind + 1)
						frontiers.push_back({});
					frontiers[ind + 1].push_back(frontiers[ind][i]->dominated[j]);
				}
			}
		}
		ind++;
	}

	counter = 0;

	vector<unsigned short> removedIndexes;
	unsigned short lastI;
	removedIndexes.reserve(2);

	for (unsigned short i = 0; true; i++) {

		if (counter + frontiers[i].size() > population.size() - 2) {

			sort(frontiers[i].begin(), frontiers[i].end(), [](Individual* i1, Individual* i2) {
				return i1->score.x < i2->score.x;
			});

			frontiers[i][0]->crowding = DBL_MAX;
			frontiers[i][frontiers[i].size() - 1]->crowding = DBL_MAX;

			for (unsigned short k = 1; k < frontiers[i].size() - 1; k++) {
				frontiers[i][k]->crowding += (frontiers[i][k + 1]->score.x - frontiers[i][k - 1]->score.x) /
					(minMaxObj1.y - minMaxObj1.x);
			}

			sort(frontiers[i].begin(), frontiers[i].end(), [](Individual* i1, Individual* i2) {
				return i1->score.y < i2->score.y;
				});

			frontiers[i][0]->crowding = DBL_MAX;
			frontiers[i][frontiers[i].size() - 1]->crowding = DBL_MAX;

			for (unsigned short k = 1; k < frontiers[i].size() - 1; k++) {
				frontiers[i][k]->crowding += (frontiers[i][k + 1]->score.y - frontiers[i][k - 1]->score.y) /
					(minMaxObj2.y - minMaxObj2.x);
			}

			sort(frontiers[i].begin(), frontiers[i].end(), [](Individual* i1, Individual* i2) {
				return i1->crowding > i2->crowding;
			});
				
			unsigned short leaveOut = counter + frontiers[i].size() - (population.size() - 2); 

			for (unsigned short j = 1; j <= leaveOut; j++) {
				removedIndexes.push_back(frontiers[i][frontiers[i].size() - j]->index);
			}

			lastI = i + 1;
			break;
		}
		else counter += frontiers[i].size();
	}

	for (unsigned short i = lastI; i < frontiers.size(); i++) {
		for (unsigned short j = 0; j < frontiers[i].size(); j++) {
			removedIndexes.push_back(frontiers[i][j]->index);
		}
	}
	
	swap(population[removedIndexes[0]], population[population.size() - 1]);
	swap(population[removedIndexes[1]], population[population.size() - 2]);
}

void WSA_EDPFSP::localSearch(MPoint<Solution*, Point<double>>& solution, unsigned int& currentNfes, bool full) {
	
	solution.y = evalSolution(solution.x, currentNfes)->score;
	
	if (full) {

		shuffle(lsFunctions.begin(), lsFunctions.end(), randGen.gen);
		for (unsigned short i = 0; i < lsFunctions.size(); i++) {
			lsFunctions[i](solution.x);
		}

		currentNfes += lsFunctions.size();
	}

	exploitation(solution);
	currentNfes++;

	solution.y = info.score;
}

void WSA_EDPFSP::ICFJIS(Solution* solution) {

	vector<unsigned short>& worstFactory = info.positions[info.worstFactoryIndex];
	vector<unsigned short>& permutation = solution->permutation;
	unsigned short randomPosition = randGen.randIntU(0, worstFactory.size() - 1);
	unsigned short removedPosition = worstFactory[randomPosition];

	worstFactory.erase(worstFactory.begin() + randomPosition);
	
	MPoint<unsigned short, double> bi = bestInsertion(solution, worstFactory, removedPosition);

	if (bi.y < info.scores[info.worstFactoryIndex].x) {

		worstFactory.insert(worstFactory.begin() + bi.x, removedPosition);
		Point<double> newScore = evalPartialSolution(solution, worstFactory);
		worstFactory.erase(worstFactory.begin() + bi.x);
		worstFactory.insert(worstFactory.begin() + randomPosition, removedPosition);
		double energyWorstFactory = info.scores[info.worstFactoryIndex].y;
		double energyGain = energyWorstFactory - newScore.y;

		if (energyGain > 0) {

			unsigned short removedJob = permutation[removedPosition];

			//update solution
			if (randomPosition < bi.x) {
				for (unsigned short i = randomPosition; i < bi.x; i++) {
					permutation[worstFactory[i]] = permutation[worstFactory[i + 1]];
				}
			}
			else {
				for (unsigned short i = randomPosition; i > bi.x; i--) {
					permutation[worstFactory[i]] = permutation[worstFactory[i - 1]];
				}
			}

			permutation[worstFactory[bi.x]] = removedJob;

			//update score

			info.scores[info.worstFactoryIndex] = newScore;

			double max = 0.;

			for (unsigned short i = 0; i < info.scores.size(); i++) {
				if (info.scores[i].x > max) {
					max = info.scores[i].x;
					info.worstFactoryIndex = i;
				}
			}

			info.score.x = info.scores[info.worstFactoryIndex].x;
			info.score.y -= energyWorstFactory - newScore.y;
		}
	}
	else {
		worstFactory.insert(worstFactory.begin() + randomPosition, removedPosition);
	}
}

void WSA_EDPFSP::ICFJS(Solution* solution) {
	vector<unsigned short>& worstFactory = info.positions[info.worstFactoryIndex];
	vector<unsigned short>& permutation = solution->permutation;

	if (worstFactory.size() < 2) return;

	unsigned int i1, i2;
	randGen.twoRandomIndexes(worstFactory.size(), i1, i2);

	unsigned short tmp = permutation[worstFactory[i1]];
	permutation[worstFactory[i1]] = permutation[worstFactory[i2]];
	permutation[worstFactory[i2]] = tmp;

	Point<double> evaluation = evalPartialSolution(solution, worstFactory);

	double energyWorstFactory = info.scores[info.worstFactoryIndex].y;
	double energyGain = energyWorstFactory - evaluation.y;

	if (evaluation.x < info.score.x && energyGain > 0) {

		info.scores[info.worstFactoryIndex] = evaluation;

		double max = 0.;

		for (unsigned short i = 0; i < info.scores.size(); i++) {
			if (info.scores[i].x > max) {
				max = info.scores[i].x;
				info.worstFactoryIndex = i;
			}
		}

		info.score.x = info.scores[info.worstFactoryIndex].x;
		info.score.y -= energyGain;
	}
	else {
		tmp = permutation[worstFactory[i1]];
		permutation[worstFactory[i1]] = permutation[worstFactory[i2]];
		permutation[worstFactory[i2]] = tmp;
	}
}


void WSA_EDPFSP::ICFJI(Solution* solution) {

	vector<unsigned short>& worstFactory = info.positions[info.worstFactoryIndex];
	vector<unsigned short>& permutation = solution->permutation;

	if (worstFactory.size() < 2) return;

	unsigned int i1, i2;
	randGen.twoRandomIndexes(worstFactory.size(), i1, i2);

	unsigned short imin = min(i1, i2);
	unsigned short imax = max(i1, i2);
	unsigned short tmp;

	while (imin < imax) {
		tmp = permutation[worstFactory[imin]];
		permutation[worstFactory[imin]] = permutation[worstFactory[imax]];
		permutation[worstFactory[imax]] = tmp;
		imin++;
		imax--;
	}

	Point<double> evaluation = evalPartialSolution(solution, worstFactory);
	double energyWorstFactory = info.scores[info.worstFactoryIndex].y;
	double energyGain = energyWorstFactory - evaluation.y;

	if (evaluation.x < info.score.x && energyGain > 0) {

		info.scores[info.worstFactoryIndex] = evaluation;

		double max = 0.;

		for (unsigned short i = 0; i < info.scores.size(); i++) {
			if (info.scores[i].x > max) {
				max = info.scores[i].x;
				info.worstFactoryIndex = i;
			}
		}

		info.score.x = info.scores[info.worstFactoryIndex].x;
		info.score.y -= energyGain;
	}
	else {

		imin = min(i1, i2);
		imax = max(i1, i2);

		while (imin < imax) {
			tmp = permutation[worstFactory[imin]];
			permutation[worstFactory[imin]] = permutation[worstFactory[imax]];
			permutation[worstFactory[imax]] = tmp;
			imin++;
			imax--;
		}
	}

}

void WSA_EDPFSP::ECFJIS(Solution* solution) {

	vector<unsigned short>& worstFactory = info.positions[info.worstFactoryIndex];
	vector<unsigned short>& permutation = solution->permutation;
	unsigned short randomPosition = randGen.randIntU(0, worstFactory.size() - 1);
	unsigned short removedPosition = worstFactory[randomPosition];

	unsigned short randomFactoryIndex = randGen.randIntU(0, info.positions.size() - 2);

	if (randomFactoryIndex >= info.worstFactoryIndex)
		randomFactoryIndex++;

	vector<unsigned short>& randomFactory = info.positions[randomFactoryIndex];

	worstFactory.erase(worstFactory.begin() + randomPosition);

	Point<double> evalWorst = evalPartialSolution(solution, worstFactory);

	if (evalWorst.x < info.scores[info.worstFactoryIndex].x) {
		MPoint<unsigned short, double> bi = bestInsertion(solution, randomFactory, removedPosition);

		randomFactory.insert(randomFactory.begin() + bi.x, removedPosition);
		Point<double> newScoreRandom = evalPartialSolution(solution, randomFactory);
		randomFactory.erase(randomFactory.begin() + bi.x);

		double energyGain = info.scores[info.worstFactoryIndex].y - evalWorst.y + 
			info.scores[randomFactoryIndex].y -newScoreRandom.y;

		if (bi.y < info.scores[info.worstFactoryIndex].x && energyGain > 0) {

			//change factory
			unsigned short removedJob = permutation[removedPosition];
			solution->factoryAssignment[removedJob] = randomFactoryIndex;

			//update permutation
			unsigned int pos = binarySearch<unsigned short>(randomFactory, removedPosition);

			randomFactory.insert(randomFactory.begin() + pos, removedPosition);

			if (pos < bi.x) {
				for (unsigned short i = pos; i < bi.x; i++) {
					permutation[randomFactory[i]] = permutation[randomFactory[i + 1]];
				}
			}
			else {
				for (unsigned short i = pos; i > bi.x; i--) {
					permutation[randomFactory[i]] = permutation[randomFactory[i - 1]];
				}
			}

			permutation[randomFactory[bi.x]] = removedJob;

			//update score

			info.scores[info.worstFactoryIndex] = evalWorst;
			info.scores[randomFactoryIndex] = newScoreRandom;

			double max = 0.;

			for (unsigned short i = 0; i < info.scores.size(); i++) {
				if (info.scores[i].x > max) {
					max = info.scores[i].x;
					info.worstFactoryIndex = i;
				}
			}

			info.score.x = info.scores[info.worstFactoryIndex].x;
			info.score.y -= energyGain;
		}
		else worstFactory.insert(worstFactory.begin() + randomPosition, removedPosition);
	}
	else worstFactory.insert(worstFactory.begin() + randomPosition, removedPosition);
}

void WSA_EDPFSP::ECFFS(Solution* solution) {

	vector<unsigned short>& worstFactory = info.positions[info.worstFactoryIndex];
	vector<unsigned short>& permutation = solution->permutation;
	unsigned short randomPosition = randGen.randIntU(0, worstFactory.size() - 1);
	unsigned short removedPosition = worstFactory[randomPosition];

	unsigned short randomFactoryIndex = randGen.randIntU(0, info.positions.size() - 2);

	if (randomFactoryIndex >= info.worstFactoryIndex)
		randomFactoryIndex++;

	vector<unsigned short>& randomFactory = info.positions[randomFactoryIndex];

	unsigned short randomIndexWorst = randGen.randIntU(0, worstFactory.size() - 1);
	unsigned short randomIndexRandom = randGen.randIntU(0, randomFactory.size() - 1);

	//swap jobs
	solution->factoryAssignment[permutation[worstFactory[randomIndexWorst]]] = randomFactoryIndex;
	solution->factoryAssignment[permutation[randomFactory[randomIndexRandom]]] = info.worstFactoryIndex;

	unsigned short tmp = permutation[worstFactory[randomIndexWorst]];
	permutation[worstFactory[randomIndexWorst]] = permutation[randomFactory[randomIndexRandom]];
	permutation[randomFactory[randomIndexRandom]] = tmp;

	Point<double> evalWorst = evalPartialSolution(solution, worstFactory);
	Point<double> newScoreRandom = evalPartialSolution(solution, randomFactory);

	double energyWorstFactory = info.scores[info.worstFactoryIndex].y;
	double energyRandomFactory = info.scores[randomFactoryIndex].y;
	double energyGain = (energyWorstFactory - evalWorst.y + energyRandomFactory - newScoreRandom.y);

	if (evalWorst.x < info.scores[info.worstFactoryIndex].x && newScoreRandom.x < info.scores[info.worstFactoryIndex].x && 
		energyGain > 0) {
		
		//update score

		info.scores[info.worstFactoryIndex] = evalWorst;
		info.scores[randomFactoryIndex] = newScoreRandom;

		double max = 0.;

		for (unsigned short i = 0; i < info.scores.size(); i++) {
			if (info.scores[i].x > max) {
				max = info.scores[i].x;
				info.worstFactoryIndex = i;
			}
		}

		info.score.x = info.scores[info.worstFactoryIndex].x;
		info.score.y -= energyGain;
	}
	else {

		//rollback 
		tmp = permutation[worstFactory[randomIndexWorst]];
		permutation[worstFactory[randomIndexWorst]] = permutation[randomFactory[randomIndexRandom]];
		permutation[randomFactory[randomIndexRandom]] = tmp;

		solution->factoryAssignment[permutation[worstFactory[randomIndexWorst]]] = info.worstFactoryIndex;
		solution->factoryAssignment[permutation[randomFactory[randomIndexRandom]]] = randomFactoryIndex;
	}

}

void WSA_EDPFSP::exploitation(MPoint<Solution*, Point<double>>& solution) {

	unsigned int tmp = 0;

	double eImprovement = 0, partialEImprovement;

	for (unsigned short i = 0; i < info.positions.size(); i++) {
		partialEImprovement = partialExploitation(solution.x, info.positions[i]);
		info.scores[i].y -= partialEImprovement;
		eImprovement += partialEImprovement;
	}

	info.score.y -= eImprovement;
}

double WSA_EDPFSP::partialExploitation(Solution* solution, vector<unsigned short>& factory) {

	unsigned short job, speed;
	double realExecutionTime, extTime, deltaTime;

	double gain = 0;

	for (unsigned short j = 0; j < instance.machines; j++) {
		for (unsigned short i = 0; i < factory.size(); i++) {
			job = solution->permutation[factory[i]];

			realExecutionTime = instance.pt[job][j] / instance.v[solution->velocities[j][job]] + instance.s[job][j];

			if (i == 0 && j == 0) {
				o[0][0].x = 0;
				o[0][0].y = realExecutionTime;
			}
			else if (i == 0 && j != 0) {
				o[0][j].x = o[0][j - 1].y;
				o[0][j].y = o[0][j].x + realExecutionTime;
			}
			else if (j == 0) {
				o[i][0].x = o[i - 1][0].y;
				o[i][0].y = o[i][0].x + realExecutionTime;
			}
			else if (j != 0) {
				o[i][j].x = max(o[i][j - 1].y, o[i - 1][j].y);
				o[i][j].y = o[i][j].x + realExecutionTime;
			}

			if (i != factory.size() - 1 && j != 0) {

				speed = solution->velocities[j - 1][job];

				if (speed > 0) {
					extTime = min(o[i + 1][j - 1].x - o[i][j - 1].y, o[i][j].x - o[i][j - 1].y);

					if (extTime != 0) {
						deltaTime = instance.pt[job][j - 1] * (1 / instance.v[speed - 1] - 1 / instance.v[speed]);
						if (extTime >= deltaTime) {

							solution->velocities[j - 1][job]--;
							gain += instance.pt[job][j - 1] * (instance.pV[speed] / instance.v[speed] - instance.pV[speed - 1] / instance.v[speed - 1]);
							gain += deltaTime * instance.ip[j - 1];
						}
					}
				}
			}
		}
	}

	return gain;
}

void WSA_EDPFSP::print(vector<MPoint<Solution*, Point<double>>>& population) {
	for (unsigned short i = 0; i < population.size(); i++) {
		population[i].x->print();
		cout << population[i].y.x << ":" << population[i].y.y << endl;
	}

	cout << endl;
}

void WSA_EDPFSP::rawPrint(vector<MPoint<Solution*, Point<double>>>& population) {
	for (unsigned short i = 0; i < population.size(); i++) {
		population[i].x->rawPrint();
		cout << population[i].y.x << ":" << population[i].y.y << endl;
	}
	cout << endl;
}