from subprocess import Popen
from subprocess import DEVNULL, STDOUT
import glob
import os
import sys
import getopt

def main(argv):
    maxProcessors = 5
    trialsPerIstance = 20
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

    processorsAvailable = maxProcessors
    procs = []

    for command in commands:
        trialsRemaining = trialsPerIstance

        for i in range(trialsRemaining):
            procs.append(Popen(command, stdout=DEVNULL, stderr=DEVNULL, shell=True))
            processorsAvailable = processorsAvailable - 1

            if processorsAvailable == 0:
                procs[0].wait()
                processorsAvailable = processorsAvailable + 1
                procs = procs[1:]

if __name__ == '__main__':
    main(sys.argv[1:])