from subprocess import Popen
from subprocess import DEVNULL, STDOUT

import multiprocessing
from multiprocessing import Pool

import glob
import os
import sys
import getopt

def worker(command):
    p = Popen(command, stderr=DEVNULL, shell=True)
    p.wait()

def main(argv):
    maxProcessors = None
    trialsPerIstance = 10
    exePath = ""
    instancesPath = ""
    funcParams = None

    try:
        opts, args = getopt.getopt(argv, "e:i:p:t:f:", ["exe=", "ipath=", "procs=", "trials=", "funcParams="])
    except getopt.GetoptError:
        print("test.py -e <exe_PATH> -i <input_PATH> [-p <N_PROCESSORS>] [-t <N_TRIALS>]")
        sys.exit(-1)

    required = ["-e", "-i"]
    optStrings = [opt[0] for opt in opts]

    for req in required:
        if req not in optStrings:
            print("test.py -e <exe_PATH> -i <input_PATH> [-p <N_PROCESSORS>] [-t <N_TRIALS>]")
            sys.exit(-1)

    for opt, arg in opts:
        if opt == '-e':
            exePath =  arg
        elif opt == "-i":
            instancesPath = arg
        elif opt == "-p":
            maxProcessors = int(arg)
        elif opt == "-t":
            trialsPerIstance = int(arg)
        elif opt == "-f":
            funcParams = arg

    maxProcessors = maxProcessors if maxProcessors is not None else multiprocessing.cpu_count()

    print("Il test verrà eseguito con i seguenti parametri:")
    print("Percorso eseguibile: {0}".format(exePath))
    print("Percorso istanze: {0}".format(instancesPath))
    print("Processori utilizzabili: {0}".format(str(maxProcessors)))
    print("Tentativi per istanza: {0}".format(str(trialsPerIstance)))
    print("Parametri aggiuntivi: {0}".format("Nessuno" if funcParams is None else funcParams))

    instances = glob.glob(os.path.join(instancesPath, "*.txt"))
    for index, instance in enumerate(instances):
        instances[index] = instance.replace("\\", "/")

    commands = None
    if funcParams is not None:
        commands = ["{0} {1} {2}".format(exePath, instance, funcParams) for instance in instances]
    else:
        commands = ["{0} {1}".format(exePath, instance) for instance in instances]


    pool = Pool(maxProcessors)
    results = []

    for command in commands:
        for i in range(trialsPerIstance):
            results.append(pool.apply_async(worker, args=(command,)))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main(sys.argv[1:])