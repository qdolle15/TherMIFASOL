import numpy as np
import sys
import os
import serial
import time
import binascii
import matplotlib.pyplot as plt
import scienceplots
import datetime

plt.style.use(['science', 'notebook', 'grid'])

file_name = "dev_pyro_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = "Montee_CN_600_1100"
os.system(f"mkdir {file_name}")
doIwrite = True
delay = 0.018  # Delay for initialization of the global settings of the pyrometer
tempo = 0.018  # Delay for the prgm

get_Temp_device_each_s = 2  # sec
'----------------------------------------------------------'
os.system("sudo setserial /dev/ttyUSB0 low_latency")

commentaire_essai = 'Décodage des octets de la bonne manière'

'----------------------------------------------------------'
'                   Fonction pyromètre                     '
'----------------------------------------------------------'
def convertHexa2Float(value):
    return int(value[2:], 16)*256 + int(value[:2], 16)

def ThreeTempInRow(_buffer, nb_packet):
    """ Retourne les températures respectivement bi-chromatique, cannal 1 et canal 2 en degrés celsius. """
    t2C = np.asarray([convertHexa2Float(_buffer[i:i+4]) for i in range(0, 12*nb_packet, 12)])
    tK1 = np.asarray([convertHexa2Float(_buffer[i:i+4]) for i in range(4, 12*nb_packet, 12)])
    tK2 = np.asarray([convertHexa2Float(_buffer[i:i+4]) for i in range(8, 12*nb_packet, 12)])

    return t2C, tK1, tK2

def laser_state(device, state:bool):
    """ Switch the state of the targeting laser. """

    device.write(b''.join([b'00la', str(int(state)).encode(), b'\r']))
    time.sleep(delay)
    return device.read(device.in_waiting) == b'ok\r'

def get_emissivity(device, channel:int):
    """ Return the emissivity of the associeted channel. """

    if channel not in [1, 2]:
        return None
    else:
        device.write(b''.join([b'00eg', str(channel).encode(), b'\r']))
        time.sleep(delay)
        return int(device.read(device.in_waiting), 16)/1000

def set_emissivity(device, channel:int, value:float):
    """ Change the emissivity of the associeted channel. """

    if value > 1.2 or value < 0.8 or channel not in [1, 2]:
        return False
    else:
        device.write(b''.join([b'00eg', str(channel).encode(), hex(int(value * 1000)).replace('x', '').encode(), b'\r']))
        time.sleep(delay)
        return device.read(device.in_waiting) == b'ok\r'

def get_K_factor(device):
    """ Return the K-factor of the device. """

    device.write(b'00eg0\r')
    time.sleep(delay)
    return int(device.read(device.in_waiting), 16)/1000

def set_K_factor(device, value:float):
    """ Set the K-factor of the device. """

    if value > 1.2 or value < 0.8:
        return False
    else:
        device.write(b''.join([b'00eg0', hex(int(value * 1000)).replace('x', '').encode(), b'\r']))
        time.sleep(delay)
        return device.read(device.in_waiting) == b'ok\r'

def get_buffer_mode(device):
    """ Return the used buffer mode :
    0 : Display temperature
    1 : AAAABBBBCCCC (2-color temp. - K1 temp. - K2 temp.)
    2 : AAAABBBBCCCCDDDDEEEEFFFFGGHHIIJJ
    3 : AAAABBBBCCCCDDDDEEEEFFFFGGHHIIJJKKKKLLLLNNNNMMMM
    see documentation for more details. """

    device.write(b'00bum\r')
    time.sleep(delay)
    return int(device.read(device.in_waiting), 16)

def set_buffer_mode(device, mode:int):
    """ Set the buffer mode :
    0 : Display temperature
    1 : AAAABBBBCCCC (2-color temp. - K1 temp. - K2 temp.)
    2 : AAAABBBBCCCCDDDDEEEEFFFFGGHHIIJJ
    3 : AAAABBBBCCCCDDDDEEEEFFFFGGHHIIJJKKKKLLLLNNNNMMMM
    see documentation for more details. """

    if mode not in [0, 1, 2, 3]:
        return False
    else:
        device.write(b''.join([b'00bum0', str(mode).encode(), b'\r']))
        time.sleep(delay)
        return device.read(device.in_waiting) == b'ok\r'

def get_filling_interval_buffer(device):
    """ Return the filling interval of the intermediate pyrometer buffers in microseconds. """

    device.write(b'00but\r')
    time.sleep(delay)
    return int(device.read(device.in_waiting), 16) * 10

def set_filling_interval_buffer(device, interval:int):
    """ Set the filling interval of the intermediate pyrometer buffers of microseconds input values. """

    device.write(b''.join([b'00but', f"{(int(np.round(interval/10))):0{4}x}".encode(), b'\r']))
    time.sleep(delay)
    if device.read(device.in_waiting) == b'ok\r':
        return True
    else:
        return False

def get_response_time(device):
    """ Return the pyrometer response time in microseconds. """

    device.write(b'00et\r')
    time.sleep(delay)
    return int(device.read(device.in_waiting), 16) * 100

def set_response_time(device, interval:float):
    """ Set the filling interval of the intermediate pyrometer buffers of microseconds input values. """

    if interval < 0 or interval > 10e6:
        return False  # out of range
    else:
        device.write(b''.join([b'00et', f"{(int(np.round(interval / 100))):0{6}x}".encode(), b'\r']))
        time.sleep(delay)
        return device.read(device.in_waiting) == b'ok\r'

def get_temperature_device(device):
    """ Return the temperature (°C) of the sensor (1 - Device). """

    device.write(b'00tsc0\r')
    time.sleep(delay)
    temp_dev = int(device.read(device.in_waiting), 16)/256  # 1/256°C

    return temp_dev

def get_temperature_detector(device):
    """ Return the temperature (°C) of the sensor (Detector). """

    device.write(b'00tsc1\r')
    time.sleep(delay)
    temp_detec = int(device.read(device.in_waiting), 16)/256  # 1/256°C

    return temp_detec

'----------------------------------------------------------'
'                 Partie initialisation                    '
'----------------------------------------------------------'
periode_write_, periode_read_, periode_post_, periode_temp_capt = [], [], [], []
nb_data, cpt, buf_overflow = 0, 0, 0
try:
    ser = serial.Serial(port='/dev/ttyUSB0', baudrate=921600, timeout=None, writeTimeout=0, stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_EVEN)
    ser.close()
    ser.open()

    connexion_check = []
    connexion_check.append(set_emissivity(ser, channel=1, value=1))
    connexion_check.append(set_emissivity(ser, channel=2, value=1))
    connexion_check.append(set_K_factor(ser, value=1))
    connexion_check.append(set_buffer_mode(ser, mode=1))
    connexion_check.append(set_filling_interval_buffer(ser, 110))
    connexion_check.append(set_response_time(ser, 0))
    if np.all(np.asarray(connexion_check)):
        ser.read(ser.in_waiting)
    else:
        doIwrite = False
        sys.exit("Failed to establish a connection with the pyrometer.")

    with open('./workspace/temp_device.txt', "w") as f1, open('./workspace/temp_scene.txt', 'w') as f2, open('./check_error.txt', "w") as f3:
        
        print("running...")

        start_time = time.perf_counter()
        time_temp_device = time.perf_counter()
        cpt_buffer_vide=0
        while True:

            running_tps = np.round(time.perf_counter() - start_time, 3)
            sys.stdout.write("\r%d compteur -- temps écoulé %.3f" % (cpt, running_tps))
            sys.stdout.flush()
        
            cpt+=1
            # Get device temperature each <get_Temp_device_each_s> secondes
            if time.perf_counter() - time_temp_device > get_Temp_device_each_s:
                t_Tdev = time.perf_counter()
                f1.write(f"{t_Tdev - start_time} {get_temperature_device(ser)}\n")
                periode_temp_capt.append(time.perf_counter() - t_Tdev)
                time_temp_device = time.perf_counter()

            t_write = time.perf_counter()
            ser.write(b'00buf\r')  # Request the buffer content
            time.sleep(delay)  # Delay for communication
            periode_write_.append(time.perf_counter() - t_write)  # duration for 'write' / start time request

            t_read = time.perf_counter()
            raw_buf = ser.read(ser.in_waiting)  # Read the buffer content
            periode_read_.append(time.perf_counter() - t_read)

            t_post = time.perf_counter()
            buf = binascii.b2a_hex(raw_buf)  # Fully convert into hexadecimal data
            f3.write(str(buf) + '\n')

            if len(buf)>0:
                data_bytes = buf[4:]
                
                if ((len(buf)-4)%12) & ((int(buf[-2:], 16)==1) | (int(buf[-2:], 16)==0)):
                    # should be equal to 2 AND equal to either 01 or 00. Nothing else
                    buf_overflow += int(buf[-2:], 16)  

                nb_packet_no_compromised = (len(buf)-4)//12  # 4 first digits, meta data
                buf_two_color, buf_K1_temp, buf_K2_temp = ThreeTempInRow(_buffer=data_bytes, nb_packet=nb_packet_no_compromised)  # Manage buffer format and convert into decimal numbers

                time_buffer = np.linspace(t_write - start_time, time.perf_counter() - start_time, nb_packet_no_compromised)
                np.savetxt(f2, np.vstack((time_buffer, buf_two_color/10, buf_K1_temp/10, buf_K2_temp/10)).T)
                nb_data += len(buf_two_color)

                periode_post_.append(time.perf_counter() - t_post)
            else:
                cpt_buffer_vide+=1
                print(f"Problem buffer n°: {cpt_buffer_vide}")
    temps_acqui = time.perf_counter() - start_time

except KeyboardInterrupt:
    print("\nExit program...")


# Report :
if doIwrite:
    try:
        temps_dev, T_device  = np.loadtxt(f"./workspace/temp_device.txt").T
        tw, tr, tp = np.asarray(periode_write_), np.asarray(periode_read_), np.asarray(periode_post_)
        t_Temp_dev = np.asarray(periode_temp_capt)

        with open("./workspace/rapport.txt", "w") as f:
            f.write(f"Date : {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write("\n-------------------\nCommentaire : \n-------------------" + str(commentaire_essai))

            f.write("\n\n-------------------\nParamètres de la communication :\n-------------------")
            f.write(f"\nPort utilisé : {ser.port}\nBaudrate : {ser.baudrate}\nParité : {ser.parity}\nStop bit : {ser.stopbits}\nTimeout : {ser.timeout}\nWrite timeout : {ser.write_timeout}\nTemporisation communication : {tempo} s")

            f.write("\n\n-------------------\nParamètres pyromètre :\n-------------------")
            f.write(f"\nFacteur K : {get_K_factor(ser)}\nEmissivité 1 : {get_emissivity(ser, 1)}\nEmissivité 2 : {get_emissivity(ser, 2)}\n")
            f.write(f"\nMode de récupération des données : {get_buffer_mode(ser)}")
            f.write(f"\nTemps de remplissage du buffer : {get_filling_interval_buffer(ser)} µs\nTemps de réponse du capteur : {get_response_time(ser)} µs")

            f.write("\n\n-------------------\nStatistiques de l'essai :\n-------------------")
            t_boucle = np.mean(tr) + np.mean(tw) + np.mean(tp)
            f.write(f"\nTemps de l'acquisition : {temps_acqui} s")
            f.write(f"\nPériode moyenne d'un cycle : {t_boucle} s (Somme des moyennes des 3 étapes de la boucle 'for' sur chaque cycle)")
            f.write(f"\nFréquence moyenne : {int(nb_data / temps_acqui)} (Nombre de données / temps de l'acquisition)")
            f.write(f"\nTemps moyen pour la partie 'ser.write' : {np.round(np.mean(tw[0,:]), 6)} soit {np.round(np.mean(tw[0,:])/t_boucle, 6) * 100} % (Comprends la temporisation)")
            f.write(f"\nTemps moyen pour la partie 'ser.read' : {np.round(np.mean(tr[:,0]), 6)} soit {np.round(np.mean(tr[:,0])/t_boucle, 6) * 100} %")
            f.write(f"\n\n(%) buffer overflow : {np.round(100*buf_overflow/cpt, 3)}%")
            f.write(f"\nNomb")

            f.write("\n\n-------------------\nTempérature du capteur :\n-------------------")
            f.write(f"\nTempérature min. : {np.min(T_device)} °C\nTempérature max. : {np.max(T_device)} °C\nTempérature moyenne : {np.mean(T_device)} °C")
            f.write(f"\n\nTemps moyen pour la lecture température du capteur : {np.round(np.mean(temps_dev), 6)} s")
    except:
        ser.close()
else:
    ser.close()


# os.system(f"mv ./workspace/* ./{file_name}/")