package de.tudarmstadt.lt.cw.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.Charset;
import java.util.ArrayList;

import de.tudarmstadt.lt.cw.graph.ArrayBackedGraph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;

public class GraphReader {
	final static Charset UTF_8 = Charset.forName("UTF-8");
	
	public static StringIndexGraphWrapper<Float> readABCIndexed(Reader r, boolean includeSelfEdges, int maxNumEdgesPerNode, float minEdgeWeight) throws IOException {
		System.out.println("Reading input graph...");
		ArrayBackedGraph<Float> g = new ArrayBackedGraph<Float>(1024 * 1024, maxNumEdgesPerNode);
		StringIndexGraphWrapper<Float> gWrapper = new StringIndexGraphWrapper<Float>(g);
		BufferedReader reader = new BufferedReader(r);
		String line;
		ArrayList<Integer> targets = new ArrayList<Integer>(maxNumEdgesPerNode);
		ArrayList<Float> weights = new ArrayList<Float>(maxNumEdgesPerNode);
		String lastNode = null;
		int numEdgesOfCurrNode = 0;
		while ((line = reader.readLine()) != null) {
			String[] lineSplits = line.split("\t");
			if (lineSplits.length < 3) {
				System.err.println("Warning: Found " + lineSplits.length + " columns instead of 3!");
				continue;
			}
			String from = lineSplits[0];
			String to = lineSplits[1];

			if (lastNode != null && !from.equals(lastNode)) {
				if (!targets.isEmpty()) {
					g.addNode(gWrapper.getIndex(lastNode), targets, weights);
					targets = new ArrayList<Integer>(maxNumEdgesPerNode);
					weights = new ArrayList<Float>(maxNumEdgesPerNode);
				}
				numEdgesOfCurrNode = 0;
			}

			lastNode = from;
			if (numEdgesOfCurrNode == maxNumEdgesPerNode) {
				continue;
			}

			float weight = Float.parseFloat(lineSplits[2]);
			if (weight >= minEdgeWeight) {
				if (includeSelfEdges || !to.equals(from)) {
					targets.add(gWrapper.getIndex(to));
					weights.add(weight);
					numEdgesOfCurrNode++;
				}
			}
		}
		
		g.addNode(gWrapper.getIndex(lastNode), targets, weights);

		System.out.println();
		return gWrapper;
	}
}
