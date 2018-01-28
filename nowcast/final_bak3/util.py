class Constant (object):
    TS = "time_series"
    DT = "data_table"
    DATA_DIR = "nowcast_inputs"
    OUTPUT_DIR = "nowcast_benchmarks"
    DATE_STR_FMT_1 = "%Y-%m-%d"
    DATE_STR_FMT_2 = "%m/%d/%Y"

def normalize_series (x):
    return np.min (x), np.ptp (x), (x - np.min(x))/np.ptp (x)

