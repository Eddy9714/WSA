# CMakeList.txt: progetto CMake per WSA_EDPFSP. Includere l'origine e definire
# qui la logica specifica del progetto.
#
cmake_minimum_required (VERSION 3.8)

# Aggiungere l'origine all'eseguibile di questo progetto.
add_executable (WSA_EDPFSP "WSA_EDPFSP.cpp" "WSA_EDPFSP.h" "Solution.h" "Solution.cpp" "Random.h" "Globals.h" "Globals.cpp" "Main.h" "Main.cpp" "Instance.h" "Instance.cpp")

# TODO: Aggiungere i test e, se necessario, installare le destinazioni.
