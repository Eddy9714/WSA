#include "Main.h"

using namespace std;

using chrono::microseconds;
using chrono::system_clock;
using chrono::duration_cast;

namespace fs = filesystem;
//namespace fs = experimental::filesystem;

int main(int argc, char* argv[])
{
	unsigned int nfes = 0;
	unsigned short populationSize = 100;
	unsigned int seed = 0;
	double alpha = 0.9;
	double beta = 0.2;

	string path = "C:/users/edu4r/desktop/test2/20_2_5.txt";

	switch (argc) {
		case 3:
			seed = stoi(argv[2]);
		case 2:
			path = argv[1];
			break;
		case 1:
			cerr << "run with params [path] [seed*]" << endl;
			//exit(-1);
	}
		
	WSA_EDPFSP executor(path);

	nfes = executor.instance.factories * executor.instance.machines * executor.instance.jobs * 100;

	auto start = chrono::high_resolution_clock::now();
	vector<MPoint<Solution*, Point<double>>> population = executor.run(populationSize, nfes, alpha, beta, seed);
	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed = end - start;
	
	int position = path.find_last_of(".");
	string pathNoExt = path.substr(0, position);

	auto time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	position = pathNoExt.find_last_of("/");
	string filename = pathNoExt.substr(position + 1);

	string reportName = pathNoExt + "/" + to_string(seed) + "-" + to_string(time) + "-wsa-report.csv";

	if (!fs::exists(pathNoExt))
		fs::create_directory(pathNoExt);
	

	ofstream file(reportName, ios_base::app);

	if (file.is_open()) {

		file << "Evaluations;Population;alpha;beta;seed;Time" << endl;
		file << nfes << ";" << populationSize << ";" << alpha << ";" << beta << ";" << seed << ";" << elapsed.count() << endl << endl;

		for (unsigned short i = 0; i < population.size(); i++) {
			file << population[i].y.x << ";";
			file << population[i].y.y << endl;
		}

		file.close();
	}

	for (unsigned short i = 0; i < population.size(); i++) {
		delete population[i].x;
	}

	return 0;
}
