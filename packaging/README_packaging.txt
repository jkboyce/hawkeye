This directory contains files used to create standalone executable versions of
Hawkeye for Mac OS X and Windows.

Packaging is done with Pyinstaller, which can be obtained via PIP or
Anaconda (I use version 3.3.1 but others will likely work).

The two .spec files in subdirectories here are used by Pyinstaller to make
the respective executables. Pyinstaller doesn't do a perfect job of
identifying dependencies, so the .spec files specify a handful of files we
add to/remove from the build. For simplicity I keep everything that
Pyinstaller *doesn't* pick up located in the directories here so the .spec
files can easily include them.

Build instructions on OS X:

1. From the Terminal, cd to the packaging/osx directory
2. Type 'pyinstaller Hawkeye_osx.spec'
3. This should create a folder 'dist' in your current directory containing
   Hawkeye.app. Everything else in 'build' and 'dist' can be deleted.

Build instructions on Windows:

1. From the Command Prompt, cd to the packaging\windows directory
2. Type 'pyinstaller Hawkeye_windows.spec'
3. This should create a directory 'dist\Hawkeye' in your current directory
   containing Hawkeye.exe and all its dependencies.
4. Additionally the Inno Setup 5 script 'Hawkeye.iss' can be executed in
   Inno Setup (free download) to create a standalone installer.
   