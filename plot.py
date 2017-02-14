import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

def main(k_plot, constant=False):
    print(k_plot)
    with open('timing.csv') as time_file:
        csv_reader = csv.reader(time_file)
        fasts = []
        medis = []
        ns = []
        first = True
        for row in csv_reader:
            if first:
                first = False
                continue
            n = int(row[0].strip())
            k = int(row[1].strip())
            fast = int(row[2].strip())
            medi = int(row[3].strip())
            if k != k_plot:
                continue
            fasts.append(fast)
            medis.append(medi)
            ns.append(n)
        if not len(fasts):
            print("error, no data for k=%s" % k_plot)
            return
        ns = np.array(ns)
        fasts = np.array(fasts)
        medis = np.array(medis)

        if constant:
            fast_c = plt.plot(ns, fasts / (ns * k_plot), '-x', label='SMAWK')
            medi_c = plt.plot(ns, medis / (ns * k_plot), '-o', label='DP')
            plt.legend(loc='upper left')
            plt.show()
        else:
            plt.plot(ns, fasts, '-x', label='SMAWK')
            plt.plot(ns, medis, '-o', label='Fast DP')
            plt.legend(loc='upper left')
            plt.show()

    return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', default=2, type=int)
    arg_parser.add_argument('--constant', default=False,
                            action='store_const', const=True)
    args = arg_parser.parse_args()
    main(args.k, args.constant)
