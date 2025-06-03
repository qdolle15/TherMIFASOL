# MIFASOL - Thermal framework for additive manufacturing analysis

MIFASOL is a research project dedicated to the thermal analysis and modeling of parts during the Directed Energy Deposition - Additive Manufacturing (DED-AM) process. It aims to provide a modular and extensible Python framework to simulate, monitor, and analyze thermal behavior during fabrication.

The core package, `TherMIFASOL`, contains code developed during Quentin Dollé’s Ph.D. thesis, and includes tools for:
- Real-time sensor monitoring
- Experimental data processing
- Thermal modeling
- Utilities and numerical tools

The project also includes:
- Experimental data acquired during multiple test campaigns
- Calibration datasets
- Technical documentation of the measurement devices

> Note: The experimental data (`data/`) is large and stored externally (e.g., local server or external disk), but may be shared upon request.

---

## Project Structure

```text
MIFASOL/
├── TherMIFASOL/
│   ├── __init__.py
│   ├── core/
│   │   ├── sensor_monitoring/
│   │   ├── data_management/
│   │   ├── models/
│   │   └── utils/
│   ├── resources/            # Literature-based data and calibration tables
│   └── notebooks/            # Jupyter notebooks for analysis and demos
├── data/                     # External, not tracked by Git
│   ├── raw data from experiences/
│   └── data for calibration/
├── documentation/           # Technical specs of devices
│   ├── device_1/
│   ├── device_2/
│   └── ...
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Install dependencies
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install the package
```bash
pip install -e .
```

### 3. Uninstall
```bash
pip uninstall TherMIFASOL
```

---

## Dependencies
| Library     | Purpose                     |
|-------------|-----------------------------|
| `numpy`     | Numerical computations       |
| `matplotlib`| Visualization                |
| `scipy`     | Scientific functions         |
| `pandas`    | Data handling                |
| ...         | ...                          |

---

## License / Citation
Distributed for academic and research purposes. Please cite the author or the corresponding publication if used.

---

## Author
Developed by Quentin Dollé during his Ph.D. thesis.  
📧 quentin.dolle@polytechnique.edu  
Feel free to open issues or contribute.
