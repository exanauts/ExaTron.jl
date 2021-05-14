# Multi-period data generation

`rt_hourlysysload_20181002_20181009.csv` contains the hourly system load data (168 entries)
from 7 AM Oct 2 to 6 AM Oct 9 in 2018.
The data was downloaded from the [ISO New England website](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/).
In the file rows starting with "D" correspond to the system load for each hour.
For example, the line with `"D","10/02/2018","07",12297.25` indicates that
the load at 7 AM on Oct 2 2018 was 12297.25.

We generate scaling factors from the file by dividing each demand by
the first demand in time.
The following command will print out scaling factors.

```bash
$ python get_load.py rt_hourlysysload_20181002_20181009.csv > scaling_factors.txt
```

We can apply these scaling factors to generate hourly loads over a week for a power network.
Assuming that data for case13659pegase is located in `./data` directory, the following
command will generate real and reactive loads for the case with names
`case13659pegase_onehour_168.Pd` and `case13659pegase_onehour_168.Qd`, respectively.

```bash
$ python gen_load.py case13659pegase scaling_factors.txt
```
