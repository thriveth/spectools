#! /usr/bin/env python
# -*- coding: utf-8 -*-

import scipy as sp

COSlines = sp.array([['01', 'Si IV', '1122.49'],
                     ['02', 'Si II', '1190.42'],
                     ['03', 'Si II', '1193.29'],
                     ['04', 'Si III', '1206.50'],
                     ['05', 'Si II', '1260.42'],
                     ['06', 'O I', '1302.17'],
                     ['07', 'Si II', '1304.37'],
                     ['08', 'C II', '1334.53'],
                     ['09', 'Si IV', '1393.76'],
                     ['10', 'Si IV', '1402.77'],
                     ['11', 'Si II', '1526.72']])

SDSSlines = sp.array(
    [
        ['01', '_Hdelta', 4102.892],
        ['02', '_Hgamma', 4341.684],
        ['03', '_Hbeta', 4862.683],
        ['04', '_Halpha', 6564.61 ],
        ['05', '_OI' , 6302.046],
        ['06', '_OII' , 3727.092],
        ['07', '_OII' , 3729.875],
        ['08', '_OIII' , 4960.295],
        ['09', '_OIII' , 5008.240],
        ['11', '_NII', 6585.28 ],
        ['12', '_NII', 6549.85 ],
        ['13', '_SII', 6718.29 ],
        ['14', '_SII', 6732.67 ],
    ]
)

MWlines = {
    'C I 1157': 1157.1857,
    'C II 1334': 1334.53,
    'C IV 1548': 1548.204,
    'C IV 1550': 1550.781, # Doublet
    'Si IV 1122': 1122.49,
    'Si II 1190': 1190.42,
    'Si II 1193': 1193.29,
    'Si II* 1194': 1194.5001,
    'Si II* 1197': 1197.3937,
    'Si II 1260': 1260.42,
    'Si II 1304': 1304.37,
    'O I* 1305': 1304.85763,
    'O I* 1306': 1304.85763,
    'Si II* 1309': 1309.2755,
    'Si II 1526': 1526.7066,
    'Si III 1206': 1206.50,
    'Si IV 1393': 1393.76,
    'Si IV 1402': 1402.77,
    'O I 1302': 1302.17,
    'O I* 1304': 1304.85763,
    'N I 1134': 1134.42,   # Triplet
    'N I 1199': 1199.5496, # Triplet!
    'N V 1238': 1238.821,
    'N V 1242': 1242.804,
    'Mg II 1240': 1240.1,  # Doublet!
    'Mg II 2796': 2796.352,
    'P II 1152': 1152.818,
    'S II 1250': 1250.584,
    'S II 1253': 1253.811,
    'Fe II 1144': 1144.9379,
    'Fe II 1608': 1608.45085,
    'Fe II 2344': 2344.21274,
    'Fe II 2374': 2374.46004,
    'Fe II 2383': 2382.76386,
    'Ni II 1317': 1317.217,
    'Ni II 1370': 1370.0769,
    'Al III 1854': 1854.716,
    'Al III 1863': 1862.790,
}

fikdict = {'Si II 1190': 0.277,
           'Si II 1193': 0.575,
           'Si II 1260': 1.22,
           'Si II 1304': 0.0928,
           'Si IV 1122': 0.807,
           'Si IV 1393': 0.513,
           'Si IV 1402': 0.255,
           'Si II 1526': 0.133}

wlsdict = {'Si IV 1122': 1122.4849,
           'Si II 1190': 1190.4158,
           'Si II 1193': 1193.2897,
           'Si III 1206': 1206.4995,
           'Si II 1260': 1260.4221,
           'O I 1302': 1302.16848,
           'Si II 1304': 1304.3702,
           'C II 1334': 1334.5323,
           'Si IV 1393': 1393.7546,
           'Si IV 1402': 1402.7697,
           'Si II 1526': 1526.7066}

lislines = ['Si II 1190', 'Si II 1193', 'Si II 1260',
            'O I 1302', 'Si II 1304', 'C II 1334']

hislines = ['Si IV 1122', 'Si IV 1393', 'Si IV 1402']

SiIIlines = ['Si II 1190', 'Si II 1193', 'Si II 1260',
             'Si II 1304',]  # 'Si II 1526']

wofflines = ['Si II 1190', 'Si III 1206', 'O I 1302', 'C II 1334']

colorlist = ['blue', 'green', 'orange', 'cyan',
             'magenta', 'yellow', 'red', 'purple', 'black']
