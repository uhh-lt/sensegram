package de.tudarmstadt.lt.cw.global;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Random;

import org.junit.Test;

public class CWGlobalTest {

	private Random random = new Random(42);

	@Test
	public void testMinimal() throws IOException {
		Reader reader = new StringReader(sim("a", "b", 1));
		StringWriter writer = new StringWriter();
		float minEdgeWeight = 0.0f;
		int N = 200;

		CWGlobal.findAndWriteClusters(reader, new BufferedWriter(writer), minEdgeWeight, N, random);
		String result = writer.toString();

		assertThat(numberOfClusters(result), is(1));
		assertThat(cluster(0, result), is("0\t2\tb, a, "));
	}

	@Test
	public void testMinimal2() throws IOException {
		Reader reader = new StringReader(
				sim("a", "b", 1) + sim("b", "a", 1) + sim("a", "c", 0.5) + sim("c", "a", 0.5) + sim("e", "f", 0.3));
		StringWriter writer = new StringWriter();
		float minEdgeWeight = 0.0f;
		int N = 200;

		CWGlobal.findAndWriteClusters(reader, new BufferedWriter(writer), minEdgeWeight, N, random);
		String result = writer.toString();

		assertThat(numberOfClusters(result), is(2));
		assertTrue(containsCluster(clusters(result), "b, a, c, "));
		assertTrue(containsCluster(clusters(result), "f, e, "));
	}

	@Test
	public void testMultipleCallsSameResult() throws IOException {
		// a-b-c-d
		Reader reader = new StringReader(sim("a", "b", 1) + sim("b", "a", 1) + sim("b", "c", 1) + sim("c", "b", 1)
				+ sim("c", "d", 1) + sim("d", "c", 1));
		float minEdgeWeight = 0.0f;
		int N = 200;

		StringWriter writer1 = new StringWriter();
		CWGlobal.findAndWriteClusters(reader, new BufferedWriter(writer1), minEdgeWeight, N, random);
		String result1 = writer1.toString();

		reader.reset();

		StringWriter writer2 = new StringWriter();
		CWGlobal.findAndWriteClusters(reader, new BufferedWriter(writer2), minEdgeWeight, N, random);
		String result2 = writer2.toString();

		assertThat(result1, is(result2));
	}

	private static String sim(String a, String b, double similarity) {
		return a + "\t" + b + "\t" + similarity + "\n";
	}

	private static int numberOfClusters(String result) {
		return clusters(result).length;
	}

	private static String cluster(int id, String result) {
		return clusters(result)[id];
	}

	private static boolean containsCluster(String[] clusters, String cluster) {
		for (String c : clusters) {
			if (c.contains(cluster)) {
				return true;
			}
		}
		return false;
	}

	private static String[] clusters(String result) {
		return result.split("\n");
	}

}
