/**
 * This file is part of the Java Machine Learning Library
 * 
 * The Java Machine Learning Library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * The Java Machine Learning Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with the Java Machine Learning Library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * Copyright (c) 2006-2012, Thomas Abeel
 * 
 * Project: http://java-ml.sourceforge.net/
 * 
 */
package net.sf.javaml.clustering.mcl;

import java.util.Locale;


public class Matrix {
	float[][] data;
	int size;
	
    public Matrix(int size) {
        data = new float[size][size];
        this.size = size;
    }

    public Matrix(float[][] x) {
    	size = x.length;
        data = x;
    }

    public Matrix(Matrix matrix) {
        this(matrix.data);
    }

    public double get(int i, int j) {
        if (i > size - 1 || j > size - 1) {
            throw new IndexOutOfBoundsException();
        }
        return data[i][j];
    }

    public float set(int i, int j, float a) {
        if (i > size - 1 || j > size - 1) {
            throw new IndexOutOfBoundsException();
        }
        float b = data[i][j];
        data[i][j] = a;
        return b;
    }

    public int size() {
        return size;
    }

    public double add(int i, int j, double a) {
        if (i > size - 1 || j > size - 1) {
            throw new IndexOutOfBoundsException();
        }
        data[i][j] += a;
        return data[i][j];
    }

    /**
     * normalise rows to rowsum
     *
     * @param rowsum for each row
     * @return vector of old row sums
     */
    public void normalise(float rowsum) {
        for (int i = 0; i < size; i++) {
            double sum = 0;
            for (int j = 0; j < size; j++) {
            	sum += data[i][j];
            }
            double invsum = 0.0;
            if (sum > 0){
                invsum = rowsum / sum;
            }

            for (int j = 0; j < size; j++) {
            	data[i][j] *= invsum;
            }
        }
    }

    /**
     * normalise by major dimension (rows)
     */
    public void normaliseRows() {
        normalise(1.0f);
    }

    /**
     * normalise by minor dimension (columns), expensive.
    public void normaliseCols() {
        double[] sums = new double[maxVLength];
        for (int row = 0; row < size(); row++) {
            for (int col = 0; col < get(row).getLength(); col++) {
                sums[col] += get(row).get(col);
            }
        }
        for (int row = 0; row < size(); row++) {
            for (int col = 0; col < get(row).getLength(); col++) {
                get(row).mult(col, 1 / sums[col]);
            }
        }
    }
    /*

    /**
     * copy the matrix and its elements
     */
    public Matrix copy() {
        return new Matrix(this);
    }

    /**
     * immutable multiply this matrix (A) with M : A * M
     *
     * @param m
     * @return matrix product
     */
    public Matrix times(Matrix m) {
        Matrix s = new Matrix(m.size);
        for (int i = 0; i < m.size; i++) {
            for (int j = 0; j < m.size; j++) {
                for (int k = 0; k < m.size; k++) {
                    double a = m.data[k][j];
                    if (a != 0.) {
                        s.data[i][j] += data[i][k] * a;
                    }
                }
            }
        }
        return s;
    }

    /**
     * mutable m2 = m .^ s
     *
     * @param s
     * @return
     */
    public void hadamardPower(double s) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
            	data[i][j] = (float)Math.pow(data[i][j], s);
            }
        }
    }

    /*
     * (non-Javadoc)
     *
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
    	sb.append("{\n");
        for (int i = 0; i < size; i++) {
        	sb.append("{");
            for (int j = 0; j < size; j++) {
            	sb.append(String.format(Locale.ENGLISH, "%.2f", data[i][j])).append("  ");
            }
        	sb.append("}\n");
        }
    	sb.append("}");
        return sb.toString();
    }

    /**
     * prune all values whose magnitude is below threshold
     */
    public void prune(double threshold) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
            	data[i][j] = data[i][j] < threshold ? 0 : data[i][j];
            }
        }
    }
}
