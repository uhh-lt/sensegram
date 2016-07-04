This is an implementation of the Chinese Whispers graph clustering algorithm. For an introduction
or if you need to reference the algorithm, use this paper:
http://wortschatz.uni-leipzig.de/~cbiemann/pub/2006/BiemannTextGraph06.pdf

This project uses the CW algorithm specifically for Word Sense Induction (WSI).

To run algorithm with different parameters:

```
run.sh <distributional-thesaurus.csv>
```

You can compile the code using Maven, and run the WSI algorithm from the command line.

Here's a quickstart guide:
```
bash
git clone https://github.com/johannessimon/chinese-whispers.git
cd chinese-whispers && mvn package shade:shade
java -cp target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI
```

You may also of course use the CW algorithm directly from your code.

For an example of how to use the WSI algorithm, compile the code as shown above and download
example data, like this word similarity graph extracted from a 120-million-lines English news
corpus taken from the JoBimText project:
http://sourceforge.net/projects/jobimtext/files/data/models/en_news120M_stanford_lemma/LMI_p1000_l200.gz

The data is formatted in _ABC_ format, meaning that each row contains an edge of the graph,
and each row contains three columns separated by a whitespace: _from_, _to_, and the _edge weight_.

Then run the WSI algorithm on the data (making sure you assign enough memory to the VM):
```
bash
java -Xms4G -Xmx4G -cp target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI
-in /path/to/LMI_p1000_l200.gz -n 100 -N 100 -out test-output.txt
```

The output (in our case test-output.txt) is then formatted as follows:
```
    word <TAB> cluster-id <TAB> cluster-label <TAB> cluster-node1 cluster-node2 ...
    word <TAB> cluster-id <TAB> cluster-label <TAB> cluster-node1 cluster-node2 ...
    ...
```

In addition, a default implementation of chinese-whispers for global clustering is available:
```
java -Xms4G -Xmx4G -cp target/chinese-whispers.jar de.tudarmstadt.lt.cw.global.CWGlobal -in /path/to/edges.gz -N 1000 -out clusters.csv.gz
```
N limits how many edges are maximum added per node when building the graph
