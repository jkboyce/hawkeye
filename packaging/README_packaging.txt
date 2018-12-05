This directory contains files used to create standalone executable versions of
Hawkeye for OSX, Windows, and Linux.

Packaging is done with Pyinstaller, which can be obtained via PIP or
Anaconda (I use version 3.3.1 but others will likely work).

The three .spec files in subdirectories here are used by Pyinstaller to make
the respective executables.

For example on OSX:

1. From the Terminal, cd to the packaging/osx directory
2. Type 'pyinstaller Hawkeye_osx.spec'
3. This should create a folder 'dist' in your current directory containing
   Hawkeye.app. Everything else in 'build' and 'dist' can be deleted.
