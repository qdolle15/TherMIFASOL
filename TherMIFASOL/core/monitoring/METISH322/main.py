from metis_pyrometer import MetisPyrometer

with MetisPyrometer(port="/dev/ttyUSB0") as pyro:
    pyro.run(duration=60)
    data = pyro.get_numpy_array()
    pyro.generate_report()