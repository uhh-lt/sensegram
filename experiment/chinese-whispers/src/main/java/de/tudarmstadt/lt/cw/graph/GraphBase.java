package de.tudarmstadt.lt.cw.graph;

import java.io.IOException;
import java.io.Writer;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang.StringUtils;

import de.tudarmstadt.lt.util.IndexUtil;
import de.tudarmstadt.lt.util.IndexUtil.Index;

/**
 * Abstract Graph class implementing a few of the Graph interface
 * methods
 */
public abstract class GraphBase<N, E> implements Graph<N, E> {
	public GraphBase() {
		super();
	}

	public void writeDot(Writer writer) throws IOException {
		writeDot(writer, IndexUtil.<N>getIdentityIndex());
	}

	public void writeDotUndirected(Writer writer) throws IOException {
		writeDotUndirected(writer, IndexUtil.<N>getIdentityIndex());
	}

	public void writeDot(Writer writer, Index<?, N> index) throws IOException {
		writer.write("digraph g {\n\toverlap=false;\n\toutputorder=edgesfirst\n\tnode[style=filled,fillcolor=white]\n");
		Iterator<N> it = iterator();
		while (it.hasNext()) {
			N node = it.next();
			writer.write("\t" + node + " [label=\"" + index.get(node) + "\"];\n");
		}
		it = iterator();
		while (it.hasNext()) {
			N node = it.next();
			Iterator<Edge<N, E>> edgeIt = getEdges(node);
			while (edgeIt.hasNext()) {
				Edge<N, E> edge = edgeIt.next();
				String weight = String.format(Locale.US, "%.5f", (Float)edge.weight / 1000.0);
				writer.write("\t" + edge.source + " -> " + node + " [weight=" + weight + "];\n");
			}
		}
		writer.write("}\n");
		return;
	}

	public void writeDotUndirected(Writer writer, Index<?, N> index) throws IOException {
		// Entry is here used as tuple class
		Set<Entry<N, N>> edges = new HashSet<Entry<N, N>>();
		writer.write("strict graph g {\n\toverlap=false;\n\toutputorder=edgesfirst\n\tnode[style=filled,fillcolor=white]\n");
		Iterator<N> it = iterator();
		while (it.hasNext()) {
			N node = it.next();
			writer.write("\t" + node + " [label=\"" + index.get(node) + "\"];\n");
		}
		it = iterator();
		while (it.hasNext()) {
			N node = it.next();
			Iterator<Edge<N, E>> edgeIt = getEdges(node);
			while (edgeIt.hasNext()) {
				Edge<N, E> edge = edgeIt.next();
				Entry<N, N> entryEdge = new SimpleEntry<N, N>(node, edge.source);
				if (!edges.contains(entryEdge)) {
					edges.add(new SimpleEntry<N, N>(node, edge.source));
					edges.add(new SimpleEntry<N, N>(edge.source, node));
					String weight = String.format(Locale.US, "%.5f", (Float)edge.weight);
					writer.write("\t" + edge.source + " -- " + node + " [weight=" + weight + "];\n");
				}
			}
		}
		writer.write("}\n");
		return;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		Iterator<N> it = iterator();
		// avoid printing too large graphs, this usually chokes e.g. Eclipse in debug mode
		int numSubgraphNodes = 1000;
		Graph<N, E> subgraph;
		if (getSize() > numSubgraphNodes) {
			ArrayList<N> subgraphNodes = new ArrayList<N>(numSubgraphNodes);
			int numNodes = 0;
			while (it.hasNext() && numNodes < numSubgraphNodes) {
				subgraphNodes.add(it.next());
				numNodes++;
			}
			subgraph = subgraph(subgraphNodes, Integer.MAX_VALUE);
			sb.append("Graph[too large, showing only first " + numSubgraphNodes + " nodes] {\n");
		} else {
			subgraph = this;
			sb.append("Graph {\n");
		}
		it = subgraph.iterator();
		while (it.hasNext()) {
			N node = it.next();
			sb.append("\t" + node + ": ");
			sb.append(StringUtils.join(subgraph.getNeighbors(node), ','));
			sb.append("\n");
		}
		sb.append("}\n");
		return sb.toString();
	}
	
	@SuppressWarnings("unchecked")
	public boolean equals(Object other) {
		if (!(other instanceof Graph<?, ?>)) {
			return false;
		}
		Graph<N, E> otherGraph = (Graph<N, E>)other;
		List<N> nodes = IteratorUtils.toList(iterator());
		List<N> nodesOther = IteratorUtils.toList(otherGraph.iterator());
		if (!nodes.containsAll(nodesOther)) {
			return false;
		}
		if (!nodesOther.containsAll(nodes)) {
			return false;
		}

		for (N node : nodes) {
			List<Edge<N, E>> edges = IteratorUtils.toList(getEdges(node));
			List<Edge<N, E>> edgesOther = IteratorUtils.toList(otherGraph.getEdges(node));
			if (!edges.containsAll(edgesOther)) {
				return false;
			}
			if (!edgesOther.containsAll(edges)) {
				return false;
			}
		}
		
		return true;
	}
}