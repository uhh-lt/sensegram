from scipy.stats import binom
from pandas import read_csv
import numpy as np
import argparse


def mcnemar_midp(b, c):
    """Compute McNemar's test using the "mid-p" variant suggested by:
    
    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
    binary matched-pairs data: Mid-p and asymptotic are better than exact 
    conditional. BMC Medical Research Methodology 13: 91.
    
    `b` is the number of observations correctly labeled by the first---but 
    not the second---system; `c` is the number of observations correctly 
    labeled by the second---but not the first---system."""

    n = b + c
    
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    chi = float(abs(b - c)**2)/n
    
    print("b = ", b)
    print("c = ", c)
    print("Exact p = ", p)
    print("Mid p = ", midp)
    print("Chi = ", chi)


def run(set1, set2):
    r1 = read_csv(set1, sep='\t', encoding='utf8',
            dtype={'predict_sense_ids': np.str, 'gold_sense_ids': np.str, 'context_id': np.str}, 
            doublequote=False, quotechar="\\u0000" )
    r2 = read_csv(set2, sep='\t', encoding='utf8',
        dtype={'predict_sense_ids': np.str, 'gold_sense_ids': np.str, 'context_id': np.str}, 
        doublequote=False, quotechar="\\u0000" )
    
    s1 = r1["correct"].values.tolist()
    s2 = r2["correct"].values.tolist()
    
    b = sum([x and not y for (x,y) in zip(s1,s2)])
    c = sum([not x and y for (x,y) in zip(s1,s2)])
    
    mcnemar_midp(b, c)


def main():
    parser = argparse.ArgumentParser(description='Compute statistical significance of predicted label sets')
    parser.add_argument('set1', help='A path to the first evaluated dataset. Format: "context_id<TAB>target<TAB>target_pos<TAB>target_position<TAB>gold_sense_ids<TAB>predict_sense_ids<TAB>golden_related<TAB>predict_related<TAB>context<TAB>smth<TAB>correct')
    parser.add_argument("set2", help="A path to the second evaluated dataset")
    
    args = parser.parse_args()

    run(args.set1, args.set2) 


if __name__ == '__main__':
    main()