package net.sf.javaml.clustering.mcl;

import org.junit.Test;

public class MarkovClusteringTest {
	@Test
	public void testRun() {
		float[][] matrix = {{0, 1, 1, 1}, {1, 0, 0, 1}, {1, 0, 0, 0}, {1, 1, 0, 0}};
		
		Matrix m = new Matrix(matrix);
		MarkovClustering2 mc = new MarkovClustering2();
		Matrix m2 = mc.run(m, 0, 2.0, 1.0, 0.00000001);
		System.out.println(m2);
	}
}
