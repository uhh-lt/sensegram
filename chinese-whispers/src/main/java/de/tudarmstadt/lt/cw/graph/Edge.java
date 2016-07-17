package de.tudarmstadt.lt.cw.graph;

public class Edge<N, E> {
	N source;
	E weight;
	
	public Edge() {
		source = null;
		weight = null;
	}
	
	public Edge(N t, E w) {
		source = t;
		weight = w;
	}
	
	public N getSource() {
		return source;
	}
	
	public E getWeight() {
		return weight;
	}
	
	@Override
	public String toString() {
		return "Edge(target=" + source.toString() + ", weight=" + weight.toString() + ")";
	}
	
	@SuppressWarnings("rawtypes")
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null || getClass() != obj.getClass())
			return false;
		Edge other = (Edge) obj;
		return source.equals(other.source) && weight.equals(other.weight);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((source == null) ? 0 : source.hashCode());
		result = prime * result + ((weight == null) ? 0 : weight.hashCode());
		return result;
	}
	/*
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Edge other = (Edge) obj;
		if (target == null) {
			if (other.target != null)
				return false;
		} else if (!target.equals(other.target))
			return false;
		if (weight == null) {
			if (other.weight != null)
				return false;
		} else if (!weight.equals(other.weight))
			return false;
		return true;
	}*/
}