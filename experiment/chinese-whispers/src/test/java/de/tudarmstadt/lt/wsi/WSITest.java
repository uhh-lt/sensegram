package de.tudarmstadt.lt.wsi;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Ignore;
import org.junit.Test;

import de.tudarmstadt.lt.cw.graph.ArrayBackedGraph;
import de.tudarmstadt.lt.cw.graph.Graph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;
import de.tudarmstadt.lt.wsi.WSI.ClusteringAlgorithm;

public class WSITest {

	@Test
	public void testIndexedCW() throws IOException {
		Graph<Integer, Float> graph = new ArrayBackedGraph<Float>(3, 2);
		StringIndexGraphWrapper<Float> graphWrapper = new StringIndexGraphWrapper<Float>(graph);
		graphWrapper.addNode("VW");
		graphWrapper.addNode("Jaguar");
		graphWrapper.addNode("Lion");
		int vw = graphWrapper.getIndex("VW");
		int jaguar = graphWrapper.getIndex("Jaguar");
		int lion = graphWrapper.getIndex("Lion");

		graphWrapper.addEdge("Jaguar", "Lion", 1.0f);
		graphWrapper.addEdge("Lion", "Jaguar", 1.0f);
		graphWrapper.addEdge("Jaguar", "VW", 1.0f);
		graphWrapper.addEdge("VW", "Jaguar", 1.0f);

		Map<Integer, Set<Integer>> vwCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> vwCluster0 = new HashSet<Integer>();
		vwCluster0.add(jaguar);
		vwCluster.put(jaguar, vwCluster0);
		Map<Integer, Set<Integer>> jaguarCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> jaguarCluster0 = new HashSet<Integer>();
		jaguarCluster0.add(lion);
		jaguarCluster.put(lion, jaguarCluster0);
		Set<Integer> jaguarCluster1 = new HashSet<Integer>();
		jaguarCluster1.add(vw);
		jaguarCluster.put(vw, jaguarCluster1);
		Map<Integer, Set<Integer>> lionCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> lionCluster0 = new HashSet<Integer>();
		lionCluster0.add(jaguar);
		lionCluster.put(jaguar, lionCluster0);

		WSI cwd = new WSI(graphWrapper, ClusteringAlgorithm.ChineseWhispers);
		assertClusterEquals(vwCluster, cwd.findSenseClusters(vw));
		assertClusterEquals(jaguarCluster, cwd.findSenseClusters(jaguar));
		assertClusterEquals(lionCluster, cwd.findSenseClusters(lion));

		graphWrapper.addEdge("Lion", "VW", 1.0f);
		graphWrapper.addEdge("VW", "Lion", 1.0f);

		jaguarCluster = new HashMap<Integer, Set<Integer>>();
		jaguarCluster0 = new HashSet<Integer>();
		jaguarCluster0.add(lion);
		jaguarCluster0.add(vw);
		jaguarCluster.put(vw, jaguarCluster0);
		assertClusterEquals(jaguarCluster, cwd.findSenseClusters(jaguar));
	}

	@Test
	@Ignore("needs fix")
	public void testIndexedMCL() throws IOException {
		Graph<Integer, Float> graph = new ArrayBackedGraph<Float>(3, 2);
		StringIndexGraphWrapper<Float> graphWrapper = new StringIndexGraphWrapper<Float>(graph);
		graphWrapper.addNode("VW");
		graphWrapper.addNode("Jaguar");
		graphWrapper.addNode("Lion");
		int vw = graphWrapper.getIndex("VW");
		int jaguar = graphWrapper.getIndex("Jaguar");
		int lion = graphWrapper.getIndex("Lion");

		graphWrapper.addEdge("Jaguar", "Lion", 1.0f);
		graphWrapper.addEdge("Lion", "Jaguar", 1.0f);
		graphWrapper.addEdge("Jaguar", "VW", 1.0f);
		graphWrapper.addEdge("VW", "Jaguar", 1.0f);

		Map<Integer, Set<Integer>> vwCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> vwCluster0 = new HashSet<Integer>();
		vwCluster0.add(jaguar);
		vwCluster.put(jaguar, vwCluster0);
		Map<Integer, Set<Integer>> jaguarCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> jaguarCluster0 = new HashSet<Integer>();
		jaguarCluster0.add(lion);
		jaguarCluster.put(lion, jaguarCluster0);
		Set<Integer> jaguarCluster1 = new HashSet<Integer>();
		jaguarCluster1.add(vw);
		jaguarCluster.put(vw, jaguarCluster1);
		Map<Integer, Set<Integer>> lionCluster = new HashMap<Integer, Set<Integer>>();
		Set<Integer> lionCluster0 = new HashSet<Integer>();
		lionCluster0.add(jaguar);
		lionCluster.put(jaguar, lionCluster0);

		WSI cwd = new WSI(graphWrapper, ClusteringAlgorithm.MarkovChainClustering);
		assertClusterEquals(vwCluster, cwd.findSenseClusters(vw));
		assertClusterEquals(jaguarCluster, cwd.findSenseClusters(jaguar));
		assertClusterEquals(lionCluster, cwd.findSenseClusters(lion));

		graphWrapper.addEdge("Lion", "VW", 1.0f);
		graphWrapper.addEdge("VW", "Lion", 1.0f);

		jaguarCluster = new HashMap<Integer, Set<Integer>>();
		jaguarCluster0 = new HashSet<Integer>();
		jaguarCluster0.add(lion);
		jaguarCluster0.add(vw);
		jaguarCluster.put(vw, jaguarCluster0);
		assertClusterEquals(jaguarCluster, cwd.findSenseClusters(jaguar));
	}

	private void assertClusterEquals(Map<Integer, Set<Integer>> a, Map<Integer, Set<Integer>> b) {
		Collection<Set<Integer>> clustersA = a.values();
		Collection<Set<Integer>> clustersB = b.values();
		assertEquals(clustersA.size(), clustersB.size());
		for (Set<Integer> cluster : clustersA) {
			assertTrue(clustersB.contains(cluster));
		}
	}
}
