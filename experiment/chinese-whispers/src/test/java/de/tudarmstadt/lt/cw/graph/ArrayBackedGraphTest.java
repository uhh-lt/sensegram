package de.tudarmstadt.lt.cw.graph;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class ArrayBackedGraphTest {

	@Test
	public void testPerformance() {
		int numNodes = 100;
		int numEdges = 100;
		long start = System.nanoTime();
		System.out.println("Instantiating graph with " + numNodes + " nodes...");
		ArrayBackedGraph<Float> g = new ArrayBackedGraph<Float>(numNodes, numEdges);
		System.out.println("Elapsed time in seconds: " + (System.nanoTime() - start) / 1000000000.0);
		System.out.println("Adding " + numNodes + " nodes with " + numEdges + " random edges per node...");
		for (int i = 0; i < numNodes; i++) {
			List<Integer> targets = new ArrayList<Integer>(numEdges);
			List<Float> weights = new ArrayList<Float>(numEdges);
			for (int j = 0; j < numEdges; j++) {
				int to = (int)(Math.random() * numNodes);
				float weight = (float)(Math.random() * 100);
				targets.add(to);
				weights.add(weight);
				g.addEdgeUndirected(i, to, weight);
			}
		}
		System.out.println("Elapsed time in seconds: " + (System.nanoTime() - start) / 1000000000.0);
	}

	@Test
	public void testSubgraph() {
		ArrayBackedGraph<Float> g1 = new ArrayBackedGraph<Float>(3, 2);
		g1.addEdge(1, 2, 1.0f);
		g1.addEdge(1, 3, 1.0f);
		g1.addEdge(2, 1, 1.0f);
		// g2 = g1 without node 3
		ArrayBackedGraph<Float> g2 = new ArrayBackedGraph<Float>(2, 2);
		g2.addEdge(1, 2, 1.0f);
		g2.addEdge(2, 1, 1.0f);
		// g4 != g2 (node 3 only contained in g4)
		ArrayBackedGraph<Float> g4 = new ArrayBackedGraph<Float>(3, 2);
		g4.addEdge(1, 2, 1.0f);
		g4.addEdge(2, 1, 1.0f);
		g4.addNode(3);
		ArrayBackedGraph<Float> g3 = new ArrayBackedGraph<Float>(1, 1);
		g3.addEdge(1, 3, 1.0f);

		List<Integer> g1Nodes = new ArrayList<Integer>();
		g1Nodes.add(1);
		g1Nodes.add(2);
		g1Nodes.add(3);
		List<Integer> g4Nodes = g1Nodes;
		List<Integer> g2Nodes = new ArrayList<Integer>();
		g2Nodes.add(1);
		g2Nodes.add(2);
		List<Integer> g3Nodes = new ArrayList<Integer>();
		g3Nodes.add(1);
		g3Nodes.add(3);
		assertEquals(g2, g1.subgraph(g2Nodes, Integer.MAX_VALUE));
		// g1.subgraph(g1Nodes, 1) should contain node 3
		assertNotEquals(g2, g1.subgraph(g1Nodes, 1));
		// as does g4
		assertEquals(g4, g1.subgraph(g1Nodes, 1));
		assertEquals(g3, g1.subgraph(g3Nodes, Integer.MAX_VALUE));
		assertEquals(g4, g4.subgraph(g4Nodes, Integer.MAX_VALUE));
		

		assertEquals(g1, g1.subgraph(g1Nodes, Integer.MAX_VALUE));
		assertEquals(g2, g2.subgraph(g2Nodes, Integer.MAX_VALUE));
		assertEquals(g3, g3.subgraph(g3Nodes, Integer.MAX_VALUE));
		assertEquals(g4, g4.subgraph(g4Nodes, Integer.MAX_VALUE));
	}

	@Test
	public void testUndirectedSubgraph() {
		ArrayBackedGraph<Float> g1 = new ArrayBackedGraph<Float>(3, 2);
		g1.addEdge(1, 2, 1.0f);
		g1.addEdge(1, 3, 1.0f);
		g1.addEdge(2, 1, 1.0f);
		ArrayBackedGraph<Float> g2 = new ArrayBackedGraph<Float>(2, 2);
		g2.addEdge(1, 2, 1.0f);
		g2.addEdge(2, 1, 1.0f);
		ArrayBackedGraph<Float> g3 = new ArrayBackedGraph<Float>(1, 2);
		g3.addEdge(1, 3, 1.0f);
		g3.addEdge(3, 1, 1.0f);
		ArrayBackedGraph<Float> g4 = new ArrayBackedGraph<Float>(3, 1);
		g4.addEdge(1, 2, 1.0f);
		g4.addEdge(2, 1, 1.0f);
		g4.addEdge(3, 1, 1.0f);
		ArrayBackedGraph<Float> g5 = new ArrayBackedGraph<Float>(4, 1);
		g5.addEdge(1, 2, 1.0f);
		g5.addEdge(2, 1, 1.0f);
		g5.addEdge(3, 1, 1.0f);
		g5.addEdge(4, 1, 1.0f);
		// This is what g5.undirectedSubgraph({1,2,3}, 2) should yield
		ArrayBackedGraph<Float> g6 = new ArrayBackedGraph<Float>(4, 2);
		g6.addEdge(1, 2, 1.0f);
		g6.addEdge(1, 3, 1.0f);
		g6.addEdge(2, 1, 1.0f);
		g6.addEdge(3, 1, 1.0f);
		g6.addEdge(4, 1, 1.0f);
		// 1->4 missing, because 1 is only allowed to have 2 outgoing edges

		List<Integer> g1Nodes = new ArrayList<Integer>();
		g1Nodes.add(1);
		g1Nodes.add(2);
		g1Nodes.add(3);
		List<Integer> g2Nodes = new ArrayList<Integer>();
		g2Nodes.add(1);
		g2Nodes.add(2);
		List<Integer> g3Nodes = new ArrayList<Integer>();
		g3Nodes.add(1);
		g3Nodes.add(3);
		List<Integer> g5Nodes = new ArrayList<Integer>();
		g5Nodes.add(1);
		g5Nodes.add(2);
		g5Nodes.add(3);
		g5Nodes.add(4);
		assertEquals(g2, g1.undirectedSubgraph(g2Nodes));
		assertEquals(g3, g1.undirectedSubgraph(g3Nodes));
	}
}
