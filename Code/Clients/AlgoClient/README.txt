CPP manual:
Code2\Clients\AlgoClient\CPP>         mkdir build
Code2\Clients\AlgoClient\CPP\build>   cd build  # for git ignore
- double click 'algoProj.sln'
- to execute after first run on VS:
Code2\Clients\AlgoClient\CPP\build> .\Debug\OTProj.exe

PY manual:
open pycharm
open project
    select Code2\Clients\AlgoClient\PY
set interpreter
got to project structure
    delete project root
    add new root MultiViewSystem\Code2
	
Description:
3d mapping via KMeans ( https://arxiv.org/abs/1903.06904 - k-Means Clustering of Lines for Big Data)
input - lines
output- find k 3d points s.t. sum is minimized:
		sum = 0
		for line in lines:
			sum+= dist of line from closest cluster(point)
		
