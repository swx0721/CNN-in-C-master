@echo off
g++ -Wall -Wextra -g3 main.cpp CNN.cpp Volumes.cpp Filters.cpp Datasets.cpp MLP.cpp -o output/main.exe
if %errorlevel% equ 0 (
    echo Compilation successful! Running program...
    output\main.exe
) else (
    echo Compilation failed!
    pause
)