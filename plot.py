import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

CLOCKS_PER_SEC = 1000000

def main(k_plot, constant=False, csvfilename='timing.csv'):
    with open(csvfilename) as time_file:
        csv_reader = csv.reader(time_file)
        fasts = []
        medis = []
        ns = []
        first = True
        headers = next(csv_reader)
        for row in csv_reader:
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
            plt.plot(ns, fasts / CLOCKS_PER_SEC, '-x', label='SMAWK')
            plt.plot(ns, medis / CLOCKS_PER_SEC, '-o', label='Fast DP')
            last_n = ns[len(medis)-1]
            step_size = 1500000
            plt.xticks(np.arange(0, (int(last_n / step_size) + 2) * step_size, step_size))
            plt.yticks(np.arange(0, (int(medis[-1]/1000000/25)+2)*25, 25))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ylabel('Seconds')
            plt.xlabel('Input size (n)')
            plt.legend(loc='upper left')
            plt.show()

    return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-k', default=2, type=int)
    arg_parser.add_argument('--constant', default=False,
                            action='store_const', const=True)
    arg_parser.add_argument('--file', default='timing.csv')
    args = arg_parser.parse_args()
    main(args.k, args.constant, args.file)
