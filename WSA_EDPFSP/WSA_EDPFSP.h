#pragma once
#include <string>
#include <vector>
#include <functional>

#include "Instance.h"
#include "Globals.h"
#include "Solution.h"

using namespace std;

class WSA_EDPFSP {

	private:
		struct Individual {
			unsigned short index;
			Solution* solution;
			Point<double> score;
			vector<Individual*> dominated;
			unsigned short dominatingCounter = 0;
			double crowding = 0.;
		};

		struct InfoFactories {
			vector<vector<unsigned short>> positions;
			vector<Point<double>> scores;
			unsigned short worstFactoryIndex;
			Point<double> score;
		};

		vector<double> c;
		vector<vector<Point<double>>> o;

		InfoFactories info;
		const Instance instance;
		vector<function<void(Solution*)>> lsFunctions;

		void initPopulation(vector<MPoint<Solution*, Point<double>>>&, unsigned short, unsigned int&);
		void randomMethod(Solution*);

		void DNEH(Solution*);

		void crossoverAndMutate(Solution*, Solution*, Point<Solution*>, double, double);
		void crossover(Solution*, Solution*, Point<Solution*>&, double);
		void pmx(vector<unsigned short>&, vector<unsigned short>&, vector<unsigned short>&,
			MPoint<Point<vector<unsigned short>>, Point<unsigned short>>&);
		void tpx(vector<unsigned short>&, vector<unsigned short>&, Point<vector<unsigned short>&>);
		void normalize(Solution*);
		void ndsa(vector<MPoint<Solution*, Point<double>>>&);

		void mutation(Solution*, double);
		void localSearch(MPoint<Solution*, Point<double>>&, unsigned int&, bool = true);
		void exploitation(MPoint<Solution*, Point<double>>&);
		double partialExploitation(Solution*, vector<unsigned short>&);

		int betterAndNearest(vector<MPoint<Solution*, Point<double>>>&, unsigned short);
		bool domination(Point<double>&, Point<double>&);

		unsigned int distance(Solution*, Solution*);
		InfoFactories* evalSolution(Solution*, unsigned int&, bool = true);
		MPoint<unsigned short, double> bestInsertion(Solution*, vector<unsigned short>&, unsigned short);
		Point<double> evalPartialSolution(Solution*, vector<unsigned short>&);

		void ICFJIS(Solution*);
		void ICFJS(Solution*);
		void ECFJIS(Solution*);
		void ECFFS(Solution*);
		void ICFJI(Solution*);

	public:
		WSA_EDPFSP(string path);

		vector<MPoint<Solution*, Point<double>>> run(unsigned short, unsigned int, double, double, unsigned int);

		void print(vector<MPoint<Solution*, Point<double>>>&);
		void rawPrint(vector<MPoint<Solution*, Point<double>>>&);
};