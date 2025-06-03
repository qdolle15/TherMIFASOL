import numpy as np

# Pyrometer's data selection 
## Time boundaries for data selection
boundaries_data_selection = {
    11:[2.11, 10],
    12:[2.78, 10],
    13:[1.88, 10],
    14:[2.65, 10],
    15:[2.20, 10],
    16:[2.73, 10],
}
## Time boundaries colors
boundaries_data_color = {
    11:0.24,
    12:0.52,
    13:0.15,
    14:0.35,
    15:0.70,
    16:0.5,
}
## Data selection colors
colors_plots_cordon = {
    11:'m',
    12:'c',
    13:'orange',
    14:'yellow',
    15:'green',
    16:'red',
}

# Loi puissance
params_linear = {
    'a': -1.15624522e-08,
    'n': 2.77021785e-01,
    'h11px': 250,  # [px]
    'h13px': 292,  # [px]
    'h11mm': 0,  # [mm]
    'h13mm': 1.6,  # [mm]
    'sigma11': 7.28,
    'sigma13': 4.88895763e+00,
    'b0': 0.735,
    'T0': 836
}

params_quadratic = {
    'a': -1.58450283e-05,
    'n': 8.53905413e-02,
    'h11px': 250,  # [px]
    'h13px': 292,  # [px]
    'h15px': 334,  # [px]
    'h11mm': 0.0,  # [mm]
    'h13mm': 1.6,  # [mm]
    'h15mm': 3.2,  # [mm]
    'sigma11': 722.992831,
    'sigma13': 590.97177228,
    'sigma15': 411.83673838,
    'b0': 0.735,
    'T0': 0
}


# beads mean information camera
params_clip = {
    11:{
        'start frame':550,
        'end frame':1500,
        'snapshot':1220,
        'crop low':5,
        'crop high':13,
        'FPS estimate':493.6,
    },
    12:{
        'start frame':0,
        'end frame':700,
        'snapshot':310,
        'crop low':5,
        'crop high':13,
        'FPS estimate':487.6,
    },
    13:{
        'start frame':0,
        'end frame':450,
        'snapshot':320,
        'crop low':5,
        'crop high':13,
        'FPS estimate':495.0,
    },
    14:{
        'start frame':1500,
        'end frame':2100,
        'snapshot':1823,
        'crop low':5,
        'crop high':13,
        'FPS estimate':491.4,
    },
    15:{
        'start frame':0,
        'end frame':550,
        'snapshot':470,
        'crop low':5,
        'crop high':13,
        'FPS estimate':497.5,
    },
    16:{
        'start frame':1200,
        'end frame':2000,
        'snapshot':1630,
        'crop low':5,
        'crop high':13,
        'FPS estimate':484.1,
    }
}

params_first_order = {
    11:{
        'tau':0.05658331345784039,
        'delay':2.32,
        },
    12:{
        'tau':0.08051930919763058,
        'delay':3,
        },
    13:{
        'tau':0.12800591741519407,
        'delay':2.32,
        },
    14:{
        'tau':0.21471762408634126,
        'delay':3,
        },
    15:{
        'tau':0.48463491693863536,
        'delay':2.32,
        },
    16:{
        'tau':0.6060073752938016,
        'delay':3,
        },
}

time_constant_parameters = [0.03057054, 0.02859343]

