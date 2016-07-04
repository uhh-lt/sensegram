package de.tudarmstadt.lt.cw.graph;

import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.Iterator;

import de.tudarmstadt.lt.util.IndexUtil.Index;

public interface Graph<N, E> extends Iterable<N>{
	
	public int getSize();

	public void addNode(N node);

	public void addEdgeUndirected(N from, N to, E weight);

	public void addEdge(N from, N to, E weight);

	public Iterator<N> getNeighbors(N node);

	public Iterator<Edge<N, E>> getEdges(N node);
	
	public E getEdge(N target, N source);
	
	public boolean hasNode(N node);

	/**
	 * Returns a non-modifiable undirected subgraph of this graph.<br>
	 * 
	 * <b>NOTE: The behaviour of this graph when nodes are added or removed is undefined!</b>
	 */
	public Graph<N, E> undirectedSubgraph(Collection<N> subgraphNodes);

	/**
	 * Returns a non-modifiable subgraph of this graph.<br>
	 * 
	 * <b>NOTE: The behaviour of this graph when nodes are added or removed is undefined!</b>
	 * 
	 * @param maxEdgesPerNode Maximum number of outgoing edges a node is allowed to have in this
	 *                        subgraph, remaining edges will not be added. Note that this will
	 *                        <i>not</i> take the top <code>maxEdgesPerNode</code> edges and pick
	 *                        only those to nodes in the subgraph, but instead will first pick
	 *                        those edges that point to nodes in the subgraph, and <i>then</i> take
	 *                        only the top <code>maxEdgePerNode</code> edges of these.
	 */
	public Graph<N, E> subgraph(Collection<N> subgraphNodes, int numEdgesPerNode);

	public void writeDot(Writer writer) throws IOException;

	public void writeDot(Writer writer, Index<?, N> index) throws IOException;

	public void writeDotUndirected(Writer writer) throws IOException;

	public void writeDotUndirected(Writer writer, Index<?, N> index) throws IOException;
}