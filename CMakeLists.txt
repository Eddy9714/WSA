# CMakeList.txt: file di progetto CMake di primo livello. Eseguire la configurazione globale
# e includere qui i sottoprogetti.
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project ("WSA_EDPFSP")

# Includere i sottoprogetti.
add_subdirectory ("WSA_EDPFSP")
