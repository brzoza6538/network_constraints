# network_constraints

Realizacja genu:

gen ma informacje o wszystkich DEMANDS 
chromosom ma informacje o wszystkich potencjalnych ścieżkach DEMAND
allel ma informacje o danej dla danej ścieżki

np.

```
ADMISSABLE_PATHS = {                    # <- gen
    "Demand_0_1" : {                    # <- chromosom
        "Path_0": int(sent_amount)      # <- allel
        .
        .
        .
        'Path_6 ': int(sent_amount)
    },
    .
    .
    .
    "Demand_10_11" : {
        "Path_0": int(sent_amount)
        .
        .
        .
    },
}
```
