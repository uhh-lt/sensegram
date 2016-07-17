package de.tudarmstadt.lt.mcl;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import de.tudarmstadt.lt.cw.CW;
import de.tudarmstadt.lt.cw.graph.ArrayBackedGraphSparseMCL;
import de.tudarmstadt.lt.cw.graph.Graph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;
import de.tudarmstadt.lt.cw.io.GraphReader;
import de.tudarmstadt.lt.util.FileUtil;
import de.tudarmstadt.lt.util.MonitoredFileReader;

public class MCLGlobal {
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
		options.addOption(OptionBuilder.withArgName("float").hasArg().withDescription("max residual").isRequired()
				.create("maxResidual"));
		options.addOption(
				OptionBuilder.withArgName("float").hasArg().withDescription("gamma").isRequired().create("gamma"));
		options.addOption(OptionBuilder.withArgName("float").hasArg().withDescription("loopGain").isRequired()
				.create("loopGain"));
		options.addOption(
				OptionBuilder.withArgName("float").hasArg().withDescription("maxZero").isRequired().create("maxZero"));
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
		float maxResidual = Float.parseFloat(cl.getOptionValue("maxResidual"));
		float gamma = Float.parseFloat(cl.getOptionValue("gamma"));
		float loopGain = Float.parseFloat(cl.getOptionValue("loopGain"));
		float maxZero = Float.parseFloat(cl.getOptionValue("maxZero"));
		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(inReader, false, N, minEdgeWeight);
		System.out.println("Running MCL sense clustering...");
		ArrayBackedGraphSparseMCL mcl = new ArrayBackedGraphSparseMCL(maxResidual, gamma, loopGain, maxZero);
		Map<Integer, Set<Integer>> clusters = mcl.findClusters(graphWrapper.getGraph());
		System.out.println("found " + clusters.size() + " clusters");
		int count = 0;
		for (Entry<Integer, Set<Integer>> cluster : clusters.entrySet()) {
			writer.write(String.valueOf(count++));
			writer.write("\t");
			int size = cluster.getValue().size();
			writer.write(String.valueOf(size));

			writer.write("\t");
			String label = graphWrapper.get(cluster.getKey());
			writer.write(label);
			writer.write(", ");
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
