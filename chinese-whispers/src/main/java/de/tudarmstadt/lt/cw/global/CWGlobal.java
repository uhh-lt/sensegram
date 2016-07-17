package de.tudarmstadt.lt.cw.global;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import de.tudarmstadt.lt.cw.CW;
import de.tudarmstadt.lt.cw.graph.Graph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;
import de.tudarmstadt.lt.cw.io.GraphReader;
import de.tudarmstadt.lt.util.FileUtil;
import de.tudarmstadt.lt.util.MonitoredFileReader;

public class CWGlobal {
	protected Graph<Integer, Float> graph;
	protected StringIndexGraphWrapper<Float> graphWrapper;
	protected CW<Integer> cw;
	protected int maxEdgesPerNode;
	protected String dotFilesOut;

	enum ClusteringAlgorithm {
		ChineseWhispers, MarkovChainClustering
	}

	@SuppressWarnings("static-access")
	public static void main(String args[]) throws IOException {
		CommandLineParser clParser = new BasicParser();
		Options options = new Options();
		options.addOption(OptionBuilder.withArgName("file").hasArg()
				.withDescription("input graph in ABC format (uncompressed or gzipped)").isRequired().create("in"));
		options.addOption(OptionBuilder.withArgName("file").hasArg()
				.withDescription("name of cluster output file (add .gz for compressed output)").isRequired()
				.create("out"));
		options.addOption(OptionBuilder.withArgName("integer").hasArg()
				.withDescription(
						"max. number of similar words to process for a given word (size of word subgraph to be clustered)")
				.isRequired().create("N"));
		CommandLine cl = null;
		try {
			cl = clParser.parse(options, args);
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("CWD", options, true);
			return;
		}
		String inFile = cl.getOptionValue("in");
		String outFile = cl.getOptionValue("out");
		Reader inReader = new MonitoredFileReader(inFile);
		BufferedWriter writer = FileUtil.createWriter(outFile);
		float minEdgeWeight = cl.hasOption("e") ? Float.parseFloat(cl.getOptionValue("e")) : 0.0f;
		int N = Integer.parseInt(cl.getOptionValue("N"));
		findAndWriteClusters(inReader, writer, minEdgeWeight, N, new Random());
	}

	protected static void findAndWriteClusters(Reader inReader, BufferedWriter writer, float minEdgeWeight, int N,
			Random random) throws IOException {
		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(inReader, false, N, minEdgeWeight);
		System.out.println("Running CW sense clustering...");
		CW<Integer> cw = new CW<Integer>(random);
		Map<Integer, Set<Integer>> clusters = cw.findClusters(graphWrapper.getGraph());
		System.out.println("found " + clusters.size() + " clusters");
		int count = 0;
		for (Entry<Integer, Set<Integer>> cluster : clusters.entrySet()) {
			writer.write(String.valueOf(count++));
			writer.write("\t");
			int size = cluster.getValue().size();
			writer.write(String.valueOf(size));
			writer.write("\t");
			for (Integer index : cluster.getValue()) {
				String label2 = graphWrapper.get(index);
				writer.write(label2);
				writer.write(", ");
			}
			writer.write("\n");
		}
		writer.close();
	}
}
