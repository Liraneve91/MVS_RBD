CPP manual windows(notice you need to have MultiViewSystem\Dependecies\opencv installed - ver 4.2.0):
Code2\Clients\CamClient\CPP>         mkdir build
Code2\Clients\CamClient\CPP\build>   cd build  # for git ignore
Code2\Clients\CamClient\CPP\build>   cmake ..
- double click 'camProj.sln'
- to execute after first run on VS:
Code2\Clients\CamClient\CPP\build> .\Debug\OTProj.exe

CPP manual linus(notice you need to have MultiViewSystem/Dependecies/opencv/opencv-4.2.0 installed):  
Code2\Clients\CamClient\CPP>         mkdir build
Code2\Clients\CamClient\CPP\build>   cd build  # for git ignore
Code2\Clients\CamClient\CPP\build>   cmake ..
Code2\Clients\CamClient\CPP\build>   make && ./camProjLinux

PY manual:
# run from terminal:
workon mvs
Code2\Clients\CamClient\PY> python CamMain.py
# edit and run from pycharm on RP and windows:
open pycharm -> open project: select Code2\Clients\CamClient\PY
windows or RP:
	set interpreter to mvs (RP)
	set interpreter to the one you have opencv installed on (windows)
go to project structure
    delete project root
    add new root: MultiViewSystem\Code2
	
Description:
server sends a task (e.g. find k ir blobs, send chessboard image ...)
rp sends desired output back
number of instances could be more than one (objective is 50 rps)
		
