from pandas import read_csv
import codecs
import argparse


LIST_SEP = ","
DDT_HEADER = "word\tcid\tcluster\tisas"
MIN_CLUSTER_SIZE = 5
VERBOSE = False


def read_ddt(ddt_fpath):
    df = read_csv(ddt_fpath, "\t", encoding='utf8', error_bad_lines=False, doublequote=False, quotechar="\u0000")
    df.word.fillna("", inplace=True)
    df.cid.fillna(-1, inplace=True)
    df.cluster.fillna("", inplace=True)
    df.isas.fillna("", inplace=True)
    return df


def make_closure(ddt_fpath, output_fpath, filtered_fpath, min_cluster_size):
    skipped_num = 0

    with codecs.open(output_fpath, "w", "utf-8") as output, codecs.open(filtered_fpath, "w", "utf-8") as filtered:
        df = read_ddt(ddt_fpath)

        print(DDT_HEADER, file=output)

        for i, row in df.iterrows():
            cluster = [related for related in row.cluster.split(LIST_SEP) if "?" not in related]
            isas = [related for related in row.isas.split(LIST_SEP) if "?" not in related]
            if len(cluster) >= min_cluster_size:
                print("%s\t%d\t%s\t%s" % (row.word, row.cid, LIST_SEP.join(cluster), LIST_SEP.join(isas)), file=output)
            else:
                skipped_num += 1
                if VERBOSE and skipped_num < 1000: print("\nSkipping cluster:", row.word, row.cid, row.cluster)
                print("%s\t%d\t%s\t%s" % (row.word, row.cid, LIST_SEP.join(cluster), LIST_SEP.join(isas)), file=filtered)

    print("Input #clusters:", i)
    print("Output #clusters:", i-skipped_num)
    print("Output:", output_fpath)


def main():
    parser = argparse.ArgumentParser(description='Make a closure DDT from a DDT.')
    parser.add_argument('ddt', help='Path to a csv file with disambiguated sense clusters "word<TAB>cid<TAB>cluster<TAB>isas".')
    parser.add_argument('-o', help='Output path. Default -- next to input file.', default="")
    parser.add_argument('-s', help='Minimum size of the cluster. Default -- %d.' % MIN_CLUSTER_SIZE, default=MIN_CLUSTER_SIZE)
    args = parser.parse_args()

    output_fpath = args.ddt + ".out" if args.o == "" else args.o
    filtered_fpath = output_fpath + ".filtered"
    print("Input sense clusters: ", args.ddt)
    print("Output path: ", output_fpath)
    print("Filtered path: ", filtered_fpath)
    print("Min. cluster size: ", args.s)
    make_closure(ddt_fpath=args.ddt, output_fpath=output_fpath, filtered_fpath=filtered_fpath,
            min_cluster_size=int(args.s))


if __name__ == '__main__':
    main()


