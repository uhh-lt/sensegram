package de.tudarmstadt.lt.cw.io;
import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.io.StringReader;

import org.apache.commons.io.FileUtils;
import org.junit.Test;

import de.tudarmstadt.lt.cw.graph.ArrayBackedGraph;
import de.tudarmstadt.lt.cw.graph.Graph;
import de.tudarmstadt.lt.cw.graph.StringIndexGraphWrapper;


public class GraphReaderTest {
	@Test
	public void testBasic() throws IOException {
		Graph<Integer, Float> testGraph = new ArrayBackedGraph<Float>(3, 2);
		StringIndexGraphWrapper<Float> testGraphIndexed = new StringIndexGraphWrapper<Float>(testGraph);
		testGraphIndexed.addNode("a");
		testGraphIndexed.addNode("b");
		testGraphIndexed.addNode("c");
		testGraphIndexed.addEdge("a", "b", 1.0f);
		testGraphIndexed.addEdge("a", "c", 1.7f);
		testGraphIndexed.addEdge("b", "c", 2.5f);
		
		File in = new File("src/test/resources/graph/test.txt");
		String input = FileUtils.readFileToString(in, "UTF-8");
		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(new StringReader(input), true, 2, 0.0f);
		assertEquals(graphWrapper, testGraphIndexed);
	}

	@Test
	public void testMaxNumNodes() throws IOException {
		Graph<Integer, Float> testGraph = new ArrayBackedGraph<Float>(3, 2);
		StringIndexGraphWrapper<Float> testGraphIndexed = new StringIndexGraphWrapper<Float>(testGraph);
		testGraphIndexed.addNode("a");
		testGraphIndexed.addNode("b");
		testGraphIndexed.addNode("c");
		testGraphIndexed.addEdge("a", "b", 1.0f);
		testGraphIndexed.addEdge("b", "c", 2.5f);
		
		File in = new File("src/test/resources/graph/test.txt");
		String input = FileUtils.readFileToString(in, "UTF-8");
		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(new StringReader(input), true, 1, 0.0f);
		assertEquals(graphWrapper, testGraphIndexed);
	}

	@Test
	public void testWeightThreshold() throws IOException {
		Graph<Integer, Float> testGraph = new ArrayBackedGraph<Float>(3, 2);
		StringIndexGraphWrapper<Float> testGraphIndexed = new StringIndexGraphWrapper<Float>(testGraph);
		testGraphIndexed.addNode("a");
		testGraphIndexed.addNode("b");
		testGraphIndexed.addNode("c");
		testGraphIndexed.addEdge("a", "c", 1.7f);
		testGraphIndexed.addEdge("b", "c", 2.5f);
		
		File in = new File("src/test/resources/graph/test.txt");
		String input = FileUtils.readFileToString(in, "UTF-8");
		StringIndexGraphWrapper<Float> graphWrapper = GraphReader.readABCIndexed(new StringReader(input), true, 2, 1.7f);
		assertEquals(graphWrapper, testGraphIndexed);
	}
}
