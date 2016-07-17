from __future__ import print_function
import argparse as ap
from os import listdir
from itertools import groupby


def read_file(file_name, directory):
    # opens a file and reads content
    # splits the content by '\n' into lists
    # splits each list by '\t'
    content = ''
    f = open(directory + file_name)
    content = map(lambda s: s.split('\t'), f.read().split('\n'))
    f.close()
    return content

# create a parser for command line argument
# and add two arguments 'Number of nearest neighbors'
# and 'Content directory'
parser = ap.ArgumentParser(description='Evaluate the output from the sense_parallel_py3.py script. There might a versioning problem with python3, contact me if in doubt.')
parser.add_argument('knn', type=int, help='Number of nearest neighbors.')
parser.add_argument('directory', type=str, help='Directory in which the output files are located.')

args = parser.parse_args()
knn = args.knn
file_names = listdir(args.directory)

# print all provided file names to validate their completeness
print('File names:')
for name in file_names:
    print(args.directory + name)

# read all provided files
content_list = map(lambda f: read_file(f, args.directory), file_names)

# join content from all files into one list
# then filter all empty elements and return
# a list of only the words (without nearest neighbors and similarity)
words = [c[0] for c in
         reduce(lambda c, l: c + l, content_list, [])
         if not c[0] == '']

# group subsequent words together and counts their appearance
evaluation = dict((k, len(list(v))) for (k, v) in groupby(words))

# print results of evaluation
print('Number of unique words: ' + str(len(set(evaluation.keys()))))

print('Words which have more than ' + str(knn) + ' nearest neighbors:')
[print(w) for (w, c) in evaluation.items() if c > knn]

print('Words which have less than ' + str(knn) + ' nearest neighbors:')
[print(w) for (w, c) in evaluation.items() if c < knn]
