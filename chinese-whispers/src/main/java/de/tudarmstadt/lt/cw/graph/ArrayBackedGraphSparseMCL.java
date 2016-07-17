package de.tudarmstadt.lt.cw.graph;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import de.tudarmstadt.lt.cw.CW;
import de.tudarmstadt.lt.util.IndexUtil.GenericIndex;
import de.tudarmstadt.lt.util.IndexUtil.Index;
import net.sf.javaml.clustering.mcl.MarkovClustering;
import net.sf.javaml.clustering.mcl.SparseMatrix;

public class ArrayBackedGraphSparseMCL extends CW<Integer> {
	final static MarkovClustering mcl = new MarkovClustering();

	float maxResidual;
	float gamma;
	float loopGain;
	float maxZero;

	public ArrayBackedGraphSparseMCL(float maxResidual, float gamma, float loopGain, float maxZero) {
		this.maxResidual = maxResidual;
		this.gamma = gamma;
		this.loopGain = loopGain;
		this.maxZero = maxZero;
	}

	@Override
	public Map<Integer, Set<Integer>> findClusters(Graph<Integer, Float> graph) {
		init(graph);

		int size = graph.getSize();
		SparseMatrix m = new SparseMatrix();
		Index<Integer, Integer> nodeIndex = new GenericIndex<Integer>();
		System.out.println("Building graph matrix");
		for (Integer node : graph) {
			int intNode = nodeIndex.getIndex(node);
			Iterator<Edge<Integer, Float>> it = graph.getEdges(node);
			while (it.hasNext()) {
				Edge<Integer, Float> edge = it.next();
				if (graph.hasNode(edge.getSource())) {
					int intTarget = nodeIndex.getIndex(edge.getSource());
					m.set(intNode, intTarget, edge.getWeight());
				} else {
					System.err.println("missing node for edge source:" + edge.getSource());
				}
			}
		}
		System.out.println("Running mcl");
		SparseMatrix res = mcl.run(m, maxResidual, gamma, loopGain, maxZero);
		nodeLabels = new HashMap<Integer, Integer>(size);
		System.out.println("=========================");
		for (Integer node : graph) {
			int intNode = nodeIndex.getIndex(node);
			double max = 0.0;
			for (int target = 0; target < res.size(); target++) {

				// find target with max
				if (res.get(intNode, target) > max)
					max = res.get(intNode, target);

				if (res.get(intNode, target) > 0.1) {
					nodeLabels.put(node, nodeIndex.get(target));
					break;
				}
			}
			System.out.print(max + " ");
		}
		System.out.println("");

		return getClusters();
	}
}
