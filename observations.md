### Observations
1. For each run data and incoming run, a single and unique tool is used.
1. Corresponding run, incoming run and metrology files share the same set of run ids. 

|  Batch ID  | No. Unique Run IDs | Run Data No. Entries | Incoming Run No. Entries | Metrology No. Entries |
|:----------:|:------------------:|:--------------------:|:------------------------:|:---------------------:|
|     1      |        225         |       2235645        |         4469164          |         11025         |
|     2      |        225         |       2231955        |         4402908          |         11025         |
|     3      |        225         |       2235450        |         4616887          |         11025         |
|     4      |        225         |       2241630        |         4477282          |         11025         |
|     6      |        225         |       2224380        |         4663135          |         11025         |
|     7      |        225         |       2248140        |         4440628          |         11025         |
|     8      |        225         |       2244840        |         4557355          |         11025         |
|     9      |        225         |       2253105        |         4500365          |         11025         |
|     10     |     **_180_**      |       1798890        |         3664211          |         8820          |
|     51     |        225         |       2233815        |         4565186          |         11025         |
|     52     |        225         |       2251365        |         4599790          |         11025         |
|     53     |        225         |       2241630        |         4456946          |         11025         |
|     54     |        225         |       2232780        |         4528655          |         11025         |
|     55     |     **_180_**      |       1795215        |         3607508          |         8820          |
|     56     |        225         |       2219490        |         4528286          |         11025         |
|     57     |        225         |       2244480        |         4318858          |         11025         |
|     58     |        225         |       2246790        |         4489500          |         11025         |
|     59     |        225         |       2249430        |         4422014          |         11025         |
|     60     |     **_180_**      |       1791720        |         3587828          |         8820          |

3. The run data files use the same set of 15 sensors ("Sensor_A" to "Sensor_O"). 
1. The incoming run data files use the same set of 41 sensors ("Sensor_1" to "Sensor_41")

### Attempting to eliminate collinearity
1. The newly created `Run Duration` column is collinear with all the other columns jointly.
1. The `Time Stamp` column is almost linearly related to the `Run Start Time` column.