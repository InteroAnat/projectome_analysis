# QC diff: old vs new inferred INS_HE

- Old side counts: {'R': 155, 'L': 105}
- New side counts: {'R': 157, 'L': 103}
- Changed neurons: 13
- Key drift: 112.swc and 114.swc changed Soma_Side L->R; multiple Neuron_Type changed to Unclassified.

## Reminder

Please continue QC on neuron type/side consistency before final analysis rerun.

## Changed rows (preview)

| NeuronID   | Soma_Side_old   | Soma_Side_new   | Neuron_Type_old   | Neuron_Type_new   | Soma_Region_old   | Soma_Region_new   |   N_Ipsilateral_old |   N_Ipsilateral_new |   N_Contralateral_old |   N_Contralateral_new |
|:-----------|:----------------|:----------------|:------------------|:------------------|:------------------|:------------------|--------------------:|--------------------:|----------------------:|----------------------:|
| 112.swc    | L               | R               | ITc               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   1 |                     0 |                     0 |
| 114.swc    | L               | R               | ITc               | Unclassified      | L-IDD5            | L-IDD5            |                   0 |                   1 |                     0 |                     0 |
| 422.swc    | L               | L               | ITi               | Unclassified      | L-IDD5            | L-IDD5            |                   0 |                   8 |                     0 |                     0 |
| 423.swc    | L               | L               | ITi               | Unclassified      | L-IDD5            | L-IDD5            |                   0 |                   5 |                     0 |                     0 |
| 438.swc    | L               | L               | CT                | CT                | L-IDM             | L-IDM             |                   0 |                   4 |                     0 |                     0 |
| 439.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   4 |                     0 |                     0 |
| 440.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   3 |                     0 |                     0 |
| 444.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   3 |                     0 |                     0 |
| 458.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   1 |                     0 |                     0 |
| 459.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   1 |                     0 |                     0 |
| 460.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   1 |                     0 |                     0 |
| 461.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   1 |                     0 |                     0 |
| 469.swc    | L               | L               | ITi               | Unclassified      | L-IDM             | L-IDM             |                   0 |                   2 |                     0 |                     0 |