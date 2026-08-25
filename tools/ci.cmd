@echo off
rem Local check runner for Windows. Same stages as tools/ci.sh.
rem
rem   tools\ci.cmd            native + search + python
rem   tools\ci.cmd native     one stage: native | search | python
rem
rem The plain-make stage is POSIX only and is not run here.

setlocal
if "%~1"=="" (set STAGE=all) else (set STAGE=%~1)
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

if /i "%STAGE%"=="native" goto native
if /i "%STAGE%"=="search" goto search
if /i "%STAGE%"=="python" goto python
if /i "%STAGE%"=="all" goto native
echo unknown stage: %STAGE% (native^|search^|python^|all)>&2
exit /b 2

:native
cmake -S native -B build\native -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || exit /b 1
cmake --build build\native --config %BUILD_TYPE% || exit /b 1
ctest --test-dir build\native -C %BUILD_TYPE% --output-on-failure || exit /b 1
if /i not "%STAGE%"=="all" goto done

:search
cmake -S search -B build\search -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || exit /b 1
cmake --build build\search --config %BUILD_TYPE% || exit /b 1
ctest --test-dir build\search -C %BUILD_TYPE% --output-on-failure || exit /b 1
if /i not "%STAGE%"=="all" goto done

:python
python -m compileall -q tools neutral || exit /b 1

:done
echo ok: %STAGE%
endlocal
