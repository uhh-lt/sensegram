package de.tudarmstadt.lt.wsi;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import de.tudarmstadt.lt.cw.CW;
import de.tudarmstadt.lt.cw.graph.ArrayBackedGraph;
import de.tudarmstadt.lt.cw.graph.ArrayBackedGraphCW;
import de.tudarmstadt.lt.cw.graph.ArrayBackedGraphMCL;
import de.tudarmstadt.lt.cw.graph.Edge;
import de.tudarmstadt.lt.cw.graph.Graph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;
import de.tudarmstadt.lt.cw.io.GraphReader;
import de.tudarmstadt.lt.util.FileUtil;
import de.tudarmstadt.lt.util.MapUtil;
import de.tudarmstadt.lt.util.MonitoredFileReader;
import de.tudarmstadt.lt.util.ProgressMonitor;

/**
 * Utility class and command-line interface for Word Sense Induction (WSI) using any graph clustering
 * algorithm (we provide implementations of Chinese Whispers and Markov Chain Clustering).
 * 
 * Run main method without arguments to see list of command line parameters.
 */
public class WSI {
	protected Graph<Integer, Float> graph;
	protected StringIndexGraphWrapper<Float> graphWrapper;
	protected CW<Integer> cw;
	protected int maxEdgesPerNode;
	protected String dotFilesOut;
	
	enum ClusteringAlgorithm {
		ChineseWhispers,
		MarkovChainClustering
	}
	
	public WSI(StringIndexGraphWrapper<Float> graphWrapper, ClusteringAlgorithm algo) {
		this(graphWrapper, Integer.MAX_VALUE, algo);
	}
	
	public WSI(StringIndexGraphWrapper<Float> graphWrapper, int maxEdgesPerNode, ClusteringAlgorithm algo) {
		this.graph = graphWrapper.getGraph();
		this.graphWrapper = graphWrapper;
		if (graph instanceof ArrayBackedGraph) {
			switch (algo) {
			case ChineseWhispers:
				ArrayBackedGraph<Float> abg = (ArrayBackedGraph<Float>)graph;
				cw = new ArrayBackedGraphCW(abg.getArraySize());
				break;
			case MarkovChainClustering:
				cw = new ArrayBackedGraphMCL(0.00000000001f, 1.4f, 0.0f, 0.0000000001f);
			}
		} else {
			cw = new CW<Integer>();
		}
		this.maxEdgesPerNode = maxEdgesPerNode;
		this.dotFilesOut = System.getProperty("wsi.debug.dotfilesout");
	}
	
	public List<Integer> getTransitiveNeighbors(Integer node, int numHops) {
		List<Integer> neighbors = new LinkedList<Integer>();
		neighbors.add(node);
		for (int i = 0; i < numHops; i++) {
			Set<Integer> _neighbors = new HashSet<Integer>();
			for (Integer neighbor : neighbors) {
				Iterator<Integer> neighborIt = graph.getNeighbors(neighbor);
				while (neighborIt.hasNext()) {
					_neighbors.add(neighborIt.next());
				}
			}
			
			neighbors.addAll(_neighbors);
		}
		
		// neighborhood excludes node itself
		neighbors.remove(node);
		return neighbors;
	}
	
	public Map<Integer, Set<Integer>> findSenseClusters(Integer node) {
		List<Integer> neighbors = getTransitiveNeighbors(node, 1);
		Graph<Integer, Float> subgraph = graph.subgraph(neighbors, maxEdgesPerNode);
		Graph<Integer, Float> undirectedSubgraph = subgraph.undirectedSubgraph(neighbors);
		
		if (dotFilesOut != null) {
			String nodeName = graphWrapper.get(node);
			try {
				String nodeNameAlphanumericOnly = nodeName.replaceAll("[^a-zA-Z\\d\\s:]", "");
				Writer graphWriter = FileUtil.createWriter(dotFilesOut + "/graph-" + nodeNameAlphanumericOnly + ".dot");
				subgraph.writeDot(graphWriter, graphWrapper);
				graphWriter.close();
				Writer graphWriter2 = FileUtil.createWriter(dotFilesOut + "/graph-" + nodeNameAlphanumericOnly + "-undirected.dot");
				undirectedSubgraph.writeDotUndirected(graphWriter2, graphWrapper);
				graphWriter2.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		return cw.findClusters(undirectedSubgraph);
	}
	
	public void findSenseClusters(Writer writer, Integer node) throws IOException {
		Map<Integer, Set<Integer>> clusters = findSenseClusters(node);
		int senseNr = 0;
		String nodeName = graphWrapper.get(node);
		for (Integer label : clusters.keySet()) {
			Set<Integer> cluster = clusters.get(label);
			String labelName = graphWrapper.get(label);
			Set<String> clusterNodeNames = new LinkedHashSet<String>();
			Map<String, Float> clusterNodeWeights = new LinkedHashMap<String, Float>();
			Iterator<Edge<Integer, Float>> edges = graphWrapper.getGraph().getEdges(node);
			while (edges.hasNext()) {
				Edge<Integer, Float> edge = edges.next();
				Integer neighbor = edge.getSource();
				if (cluster.contains(neighbor)) {
					float weight = edge.getWeight();
					String clusterNodeName = graphWrapper.get(neighbor);
					clusterNodeNames.add(clusterNodeName);
					clusterNodeWeights.put(clusterNodeName, weight);
				}
			}
			Cluster<String> c = new Cluster<String>(nodeName, senseNr, labelName, clusterNodeWeights);
			ClusterReaderWriter.writeCluster(writer, c);
			senseNr++;
		}
	}
	
	public void findSenseClusters(Writer writer) throws IOException {
		Iterator<Integer> nodeIt = graph.iterator();
		int processedNodes = 0;
		ProgressMonitor monitor = new ProgressMonitor("word graph", "nodes", graph.getSize(), 0.01);
		while (nodeIt.hasNext()) {
			Integer node = nodeIt.next();
			findSenseClusters(writer, node);
			processedNodes++;
			monitor.reportProgress(processedNodes);
		}
	}
	
	@SuppressWarnings("static-access")
	public static void main(String args[]) throws IOException {
		CommandLineParser clParser = new BasicParser();
		Options options = new Options();
		options.addOption(OptionBuilder.withArgName("file")
		                  .hasArg()
                          .withDescription("input graph in ABC format (uncompressed or gzipped)")
                          .isRequired()
                          .create("in"));
		options.addOption(OptionBuilder.withArgName("file")
		                  .hasArg()
                          .withDescription("name of cluster output file (add .gz for compressed output)")
                          .isRequired()
                          .create("out"));
		options.addOption(OptionBuilder.withArgName("integer")
		                  .hasArg()
		                  .withDescription("max. number of edges to process for each similar word (word subgraph connectivity)")
		                  .isRequired()
		                  .create("n"));
		options.addOption(OptionBuilder.withArgName("integer")
		                  .hasArg()
                          .withDescription("max. number of similar words to process for a given word (size of word subgraph to be clustered)")
                          .isRequired()
                          .create("N"));
		options.addOption(OptionBuilder.withArgName("float")
		                  .hasArg()
                          .withDescription("min. edge weight")
                          .create("e"));
		options.addOption(OptionBuilder.withArgName("float")
                .hasArg()
                .withDescription("MCL only: expansion/inflation exponent")
                .create("gamma"));
		options.addOption(OptionBuilder.withArgName("float")
                .hasArg()
                .withDescription("MCL only: loopGain (edge weight to add on self-loops, independent of their previous existence)")
                .create("loopGain"));
		options.addOption(OptionBuilder.withArgName("float")
                .hasArg()
                .withDescription("MCL only: maxResidual (tells MCL when to stop)")
                .create("maxResidual"));
		options.addOption(OptionBuilder.withArgName("float")
                .hasArg()
                .withDescription("MCL only: maxZero (max. value considered to be zero for pruning)")
                .create("maxZero"));
		options.addOption(OptionBuilder.withArgName("cw|mcl")
                .hasArg()
                .isRequired()
                .withDescription("Clustering algorithm to use: \"cw\" or \"mcl\"")
                .create("clustering"));
		options.addOption(OptionBuilder.withArgName("node-list")
                .hasArg()
                .withDescription("file with newline-separated list of nodes to cluster")
                .create("nodes"));
		CommandLine cl = null;
		boolean success = false;
		try {
			cl = clParser.parse(options, args);
			success = true;
		} catch (ParseException e) {
			System.out.println(e.getMessage());
		}

		if (!success) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("CWD", options, true);
			System.exit(1);
		}
		Set<String> nodes = cl.hasOption("nodes") ? MapUtil.readSetFromFile(cl.getOptionValue("nodes")) : null;
		String inFile = cl.getOptionValue("in");
		String outFile = cl.getOptionValue("out");
		Reader inReader = new MonitoredFileReader(inFile);
		Writer writer = FileUtil.createWriter(outFile);
		float minEdgeWeight = cl.hasOption("e") ? Float.parseFloat(cl.getOptionValue("e")) : 0.0f;
		int N = Integer.parseInt(cl.getOptionValue("N"));
		int n = Integer.parseInt(cl.getOptionValue("n"));


		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(inReader, false, N, minEdgeWeight);
		WSI cwd = null;
		if (cl.getOptionValue("clustering").toLowerCase().equals("cw")) {
			cwd = new WSI(graphWrapper, n, ClusteringAlgorithm.ChineseWhispers);
		} else if (cl.getOptionValue("clustering").toLowerCase().equals("mcl")) {
			cwd = new WSI(graphWrapper, n, ClusteringAlgorithm.MarkovChainClustering);
		} else {
			System.err.println("Unknown clustering algorithm! Must be either \"cw\" or \"mcl\"");
			return;
		}

		System.out.println("Running CW sense clustering...");
		if (nodes != null) {
			for (String node : nodes) {
				node = node.trim();
				Integer nodeIndex = graphWrapper.getIndex(node);
				cwd.findSenseClusters(writer, nodeIndex);
			}
		} else {
			cwd.findSenseClusters(writer);
		}
		writer.close();
	}
}
