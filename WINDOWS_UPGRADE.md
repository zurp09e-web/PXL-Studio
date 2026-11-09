# Updating PXL Studio on Windows to version 7.0

PXL Studio 7.0 keeps native C ABI 4, the same ABI used by PXL Studio 6.x. A
working ABI-4 DLL can be reused. A DLL from PXL 5.0 or earlier uses ABI 3 and
must be rebuilt; if no DLL is present, the Python fallback remains available.

1. Close PXL Studio and any Python process that may have loaded
   `pxl_native.dll`.
2. Extract the PXL Studio 7.0 archive into a new folder.
3. Open Command Prompt or PowerShell in the extracted project folder.
4. Install the requirements and launch the application:

   ```powershell
   py -m pip install -r requirements.txt
   py build_native.py
   py pxl_gui.py
   ```

The build script creates `pxl_native.dll` when a compatible C++17 compiler is
available (MSVC `cl.exe` or MinGW-w64 `g++`). If the linker reports that it
cannot open the DLL, close applications that may have loaded it and retry. Do
not paste compiler output back into the command prompt; only type the commands
shown above.

The new v7 context-mixing and variable-block candidate code runs in Python;
the native library continues to accelerate the existing filtering and entropy
paths. The native DLL does not need a new ABI for those Python-side additions.
