# HEStart.py
#
# Hawkeye application main entry point.
#
# Copyright 2019 Jack Boyce (jboyce@gmail.com)

import sys
from PySide2.QtWidgets import QApplication

from HEMainWindow import HEMainWindow


if __name__ == '__main__':
    profile = False

    app = QApplication(sys.argv)

    # manage our own application quit so we can properly shut
    # down the worker thread
    app.setQuitOnLastWindowClosed(False)

    window = HEMainWindow(app)
    window.show()
    window.raise_()
    app.setActiveWindow(window)

    if profile:
        import cProfile

        cProfile.run('app.exec_()')
        sys.exit()
    else:
        sys.exit(app.exec_())
