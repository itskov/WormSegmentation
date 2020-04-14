import PySimpleGUI as sg
import json
import os

from time import time


#locations = ['/home/itskov/exp1.npy', '/home/itskov/exp2.npy']
locations = os.environ['NAUTILUS_SCRIPT_SELECTED_FILE_PATHS'].splitlines()

# Here we store the json
output_dir = os.path.commonpath(locations)


sg.theme('DarkAmber')   # Add a touch of color

# Fields
Strains = ('Strain', ['TPH1', 'TPH1;SER1', 'TPH1;SER4', 'TPH1;SER5', 'TPH1;SER7', 'TPH1;MOD1', 'TPH1;MOD5'])
ExpType = ('ExpType', ['Pair Comparison', 'Multiple Experiments', 'Single Experiment'])
Proj = ('MiniProj', ['Simple', 'Time Lapse', 'Pulse Analysis'])



# All the stuff inside your window.
layout = [[sg.Text('Prepare Experiments Bundle:')],
          [sg.Text(Strains[0] + ":"), sg.Combo(Strains[1], key=Strains[0])],
          [sg.Text(ExpType[0] + ":"), sg.Combo(ExpType[1], key=ExpType[0])],
          [sg.Text(Proj[0] + ":"), sg.Combo(Proj[1], key=Proj[0])],
          [sg.Text(os.linesep.join(locations))],
          [sg.Button('Ok'), sg.Button('Cancel')]]

# Create the Window
window = sg.Window('Window Title', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):   # if user closes window or clicks cancel
        break

    if event in ('Ok'):
        print(values)
        values['files'] = locations
        jsonData = (json.dumps(values))
        print('Saving to: %s' % (output_dir,))
        with open(os.path.join(output_dir,'expBundle_' + str(int(time())) + '.json'), 'w') as f:
            f.write(jsonData)

        break



window.close()
