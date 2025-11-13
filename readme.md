# Numerical Validation 

![Numvalkit Overview](./docs/numvalkit.png)

## Setup NumValKit

```
cd NumValKit
python.exe -m pip install -e .
```

## Data Loader

### Loaders

- BatteryLoader: Load battery sequence as a List
    - load(user, with_thermal=True): also include thermal data
        `(start_time, end_time, battery_usage, thermal, charging)`
    - load(user): w/o thermal data
        `(start_time, end_time, battery_usage, charging)`

- ForegroundLoader: Load foreground application sequence as a List
    - load(user, pure_app=True): exclude desktop and off screen
    - load(user): w/ desktop and off screen
    - both return `(start, end, pkg)`

- ProductLoader: Load phone type
    - Should place a `phone_type.csv` under DATA_DIR
    - load(user): return the product type of the user

#### Generators

- BehaveVectorGenerator: Generate behavior vector
    - generate(user): default, generate category
    - generate_category(user, vector_granularity(opt), reflush(opt)): category sequence with specified granularity. reflush = True: regenerate and store the result
        return Dict of {time_interval -> (behavior vector, discharge, is_charging)}
    - generate_appname(user, vector_granularity): TODO

- ThermalSessionGenerator: Generate screen on session
    - generate(user): generate screen on session (from screen off to screen off), with minimal duration > `.minimal_duration_in_min` and start shell temp < `.thermal_start_threshold`, with its features
        - `high_temp` remarks those max temp > `thermal_overheat_threshold`

## Predictors

- BatteryAospPredictor
    Battery prediction used in AOSP

- BatteryStatisticalPredictor
    Battery prediction using statistical data (pure history)

- BatteryKmeansPredictor
    Battery prediction by matching current usage to the history

- BehaveGGNNPredictor
    Next app prediction using GGNN

- BehaveKmeansPredictor
    Next app prediction by matching current usage to the history

- BehaveTopKPredictor
    Next app prediction using the statistical topK app at 1 hour time period

- ThermalLGBMPredictor
    Thermal prediction using LGBM classifier

- ThermalRFPredictor
    Thermal prediction using Random forest

## Utils

- get_category_from_package, by white list and app_type from AOSP

## Verify Scenarios

Pure prediction:
- Next app prediction: next_app_prediction.py
- Thermal prediction: thermal_prediction.py
- Discharge prediction: discharge_prediction.py

Case simulation:
- App eviction simulation: app_eviction_simulation.py
- Extend_battery_endurance: extend_endurance_simulation.py
