package de.tudarmstadt.lt.cw;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import de.tudarmstadt.lt.cw.graph.ArrayBackedGraph;
import de.tudarmstadt.lt.cw.graph.Graph;


public class CWTest {

	@Test
	public void test() {
		Graph<Integer, Float> g = new ArrayBackedGraph<Float>(6, 6);
		int vw = 0;
		int lion = 1;
		int scary = 2;
		int hunt_a = 3;
		int drive_a = 4;
		int my = 5;
		g.addNode(vw);
		g.addNode(lion);
		g.addNode(scary);
		g.addNode(hunt_a);
		g.addNode(drive_a);
		g.addNode(my);
		
		g.addEdgeUndirected(lion, scary, 1.0f);
		g.addEdgeUndirected(lion, hunt_a, 1.0f);
		g.addEdgeUndirected(vw, drive_a, 1.0f);
		g.addEdgeUndirected(vw, my, 1.0f);
		
		Set<Integer> cluster1 = new HashSet<Integer>();
		cluster1.add(lion);
		cluster1.add(scary);
		cluster1.add(hunt_a);
		
		Set<Integer> cluster2 = new HashSet<Integer>();
		cluster2.add(vw);
		cluster2.add(drive_a);
		cluster2.add(my);
		
		CW<Integer> cw = new CW<Integer>();
		Map<Integer, Set<Integer>> clusters = cw.findClusters(g);
		assertEquals(2, clusters.size());
		assertTrue(clusters.containsValue(cluster1));
		assertTrue(clusters.containsValue(cluster2));
	}

}
