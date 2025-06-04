# First Light Imaging SDK - Installation Instructions

This section describes how to install and configure the First Light Imaging SDK (`FliSdk_V2`) required to control specific thermal camera systems in the `TherMIFASOL` framework.

---

## 1. Download the SDK

Please download the SDK installer for your operating system from the official First Light Imaging website:

https://...

---

## 2. Install the SDK

1. Run the installer and follow the instructions.
2. The Python SDK libraries will typically be installed in:

```
C:\Program Files\FirstLightImaging\FliSdk\Python\lib
```

---

## 3. Verify the installation

After installation, you should find the following directory:

```
C:\Program Files\FirstLightImaging\FliSdk\Python\lib\FliSdk_V2.py
```

---

## 4. Integration in your Python code

The SDK path is automatically added at runtime in `TherMIFASOL` with a fallback mechanism.

If the SDK is installed in the default path, nothing needs to be changed.

If it is installed elsewhere, you can set an environment variable `FLI_SDK_PATH` like this:

### On Windows (cmd):
```cmd
set FLI_SDK_PATH=C:\Your\Custom\Path\To\FliSdk\Python\lib
```

### On Linux/macOS (bash):
```bash
export FLI_SDK_PATH=/your/custom/path/to/FliSdk/Python/lib
```

Then, the code will resolve the path automatically.

---

## Troubleshooting

If the module cannot be imported, check that:
- The path is correctly set
- `FliSdk_V2.py` is present in the directory
- Python and/or Ubuntu version matches the SDK version

---

## Contact

For support, please refer to the First Light Imaging support or contact the developer of this framework.

