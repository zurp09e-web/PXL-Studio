# Windows/macOS double-click launcher; .pyw avoids opening a console window on Windows.
from multiprocessing import freeze_support
from pxl_gui import main

if __name__ == "__main__":
    freeze_support()
    main()
