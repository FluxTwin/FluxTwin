# utils/device_catalog.py
"""
Built-in device catalog with per-industry defaults.
Each device has:
- avg_kw: average power draw (kW)
- controllable: can we shift/curtail?
- shift_window_mins: typical shift flexibility
- actions: list of concrete control actions we can recommend
"""

CATALOG = {
    "Office": {
        "Workstations": {"avg_kw": 0.10, "controllable": True, "shift_window_mins": 60,
                         "actions": [
                             "Enable sleep after 10 min idle",
                             "Batch OS updates 02:00–04:00",
                             "Move heavy renders off-peak"
                         ]},
        "Server rack": {"avg_kw": 2.00, "controllable": True, "shift_window_mins": 120,
                        "actions": [
                            "Schedule backups at 02:00–05:00",
                            "Lower noncritical VM CPU limits at peak",
                            "Enable staggered reboots off-peak"
                        ]},
        "HVAC zone": {"avg_kw": 3.50, "controllable": True, "shift_window_mins": 30,
                      "actions": [
                          "Raise cooling setpoint +1°C during peak",
                          "Pre-cool 11:00–13:00 when PV surplus exists",
                          "Turn off empty rooms after 18:00"
                      ]},
        "Lighting": {"avg_kw": 0.30, "controllable": True, "shift_window_mins": 0,
                     "actions": [
                         "Enable occupancy sensors",
                         "Daylight dimming 20% 11:00–16:00",
                     ]},
    },
    "Hotel": {
        "Laundry": {"avg_kw": 4.00, "controllable": True, "shift_window_mins": 180,
                    "actions": [
                        "Run washers/dryers after 22:00",
                        "Pre-heat water 11:00–15:00 (PV)",
                        "Batch linen cycles off-peak"
                    ]},
        "Kitchen hoods": {"avg_kw": 1.50, "controllable": True, "shift_window_mins": 15,
                          "actions": [
                              "Interlock with cooktops only",
                              "Reduce fan speed 20% off-peak",
                          ]},
        "Pool boiler": {"avg_kw": 6.00, "controllable": True, "shift_window_mins": 90,
                        "actions": [
                            "Heat between 11:00–15:00 (PV)",
                            "Cover pool at night to reduce losses"
                        ]},
        "Corridor HVAC": {"avg_kw": 2.80, "controllable": True, "shift_window_mins": 30,
                          "actions": [
                              "Temperature band ±1°C at peak",
                              "Night setback after 00:00"
                          ]},
    },
    "Factory": {
        "Air compressor": {"avg_kw": 7.00, "controllable": True, "shift_window_mins": 30,
                           "actions": [
                               "Cascade setpoints across compressors",
                               "Leak check weekly; fix >5% losses",
                               "Stagger starts 15 min"
                           ]},
        "CNC line": {"avg_kw": 5.00, "controllable": True, "shift_window_mins": 60,
                     "actions": [
                         "Shift low-priority jobs after 21:00",
                         "Warm-up cycle off-peak",
                     ]},
        "Oven": {"avg_kw": 12.00, "controllable": True, "shift_window_mins": 45,
                 "actions": [
                     "Pre-heat on PV surplus",
                     "Batch jobs to avoid cycling at peak"
                 ]},
        "Lighting": {"avg_kw": 0.40, "controllable": True, "shift_window_mins": 0,
                     "actions": [
                         "Zone-based occupancy control",
                         "20% dim during daylight"
                     ]},
    },
    "Household": {
        "Water heater": {"avg_kw": 2.50, "controllable": True, "shift_window_mins": 90,
                         "actions": [
                             "Heat water 11:00–15:00 (PV)",
                             "Night off after 23:00"
                         ]},
        "EV charger": {"avg_kw": 7.40, "controllable": True, "shift_window_mins": 240,
                       "actions": [
                           "Charge after 23:00",
                           "PV-follow charging 11:00–15:00"
                       ]},
        "Dishwasher": {"avg_kw": 1.50, "controllable": True, "shift_window_mins": 120,
                       "actions": [
                           "Run after 22:00",
                           "Avoid 18:00–22:00"
                       ]},
        "AC unit": {"avg_kw": 1.20, "controllable": True, "shift_window_mins": 15,
                    "actions": [
                        "Raise setpoint +1°C at peak",
                        "Pre-cool before peak by 30 min"
                    ]},
    },
}
