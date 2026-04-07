/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.continuallearning.projection;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;

import java.io.IOException;
import java.util.Random;

/**
 * Johnson–Lindenstrauss (JL) random projection for embedding-space dimensionality
 * reduction in ContLoRA's coreset pipeline.
 *
 * <p>A single sparse Rademacher projection matrix Π ∈ ℝ^{d' × d} is generated at
 * initialisation and shared across all continual-learning stages.  Each entry is
 * drawn independently from the distribution
 * <pre>
 *   Π_{ij} = { +√(3/d') with prob 1/6,
 *             { -√(3/d') with prob 1/6,
 *             {  0        with prob 2/3
 * </pre>
 * (Achlioptas 2003 sparse construction, sparsity s = 1/3).  This guarantees the
 * JL distance-preservation property while enabling {@code O(nnz(Π) · n)} projection
 * cost rather than {@code O(d' · d · n)}.
 *
 * <p>After projection all subsequent coreset operations (build, novelty detection,
 * negative sampling) operate in the reduced {@code d'}-dimensional space.  The
 * speedup is {@code O(d/d')}, typically 4–16× for {@code d = 2048}, {@code d' = 128–512}.
 *
 * <p>The matrix is serialisable so that the same projection is used consistently
 * across node restarts.
 */
public class JLProjection implements Writeable {

    private final int inputDim;
    private final int outputDim;
    /**
     * Sparse projection matrix stored in a compact COO-like format.
     * {@code values[i]} is a list of (column-index, value) pairs for row {@code i}.
     * We store as parallel arrays {@code colIndices[i]} and {@code entryValues[i]}.
     */
    private final int[][] colIndices;
    private final float[][] entryValues;

    /**
     * Constructs a JL projection matrix with sparse Rademacher entries.
     *
     * @param inputDim  original embedding dimension {@code d}
     * @param outputDim target dimension {@code d'}, typically {@code O(log(n) / ε²)}
     * @param rng       random source for reproducibility
     */
    public JLProjection(int inputDim, int outputDim, Random rng) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        // Build sparse Rademacher matrix
        // Each entry non-zero with probability 1/3 (sparsity s=1/3 per Achlioptas 2003)
        float scale = (float) Math.sqrt(3.0 / outputDim);
        colIndices = new int[outputDim][];
        entryValues = new float[outputDim][];
        for (int row = 0; row < outputDim; row++) {
            // Count non-zeros for this row first
            int[] tempCols = new int[inputDim];
            float[] tempVals = new float[inputDim];
            int nnz = 0;
            for (int col = 0; col < inputDim; col++) {
                double u = rng.nextDouble();
                if (u < 1.0 / 6.0) {
                    tempCols[nnz] = col;
                    tempVals[nnz] = scale;
                    nnz++;
                } else if (u < 2.0 / 6.0) {
                    tempCols[nnz] = col;
                    tempVals[nnz] = -scale;
                    nnz++;
                }
                // else zero entry: skip
            }
            colIndices[row] = new int[nnz];
            entryValues[row] = new float[nnz];
            System.arraycopy(tempCols, 0, colIndices[row], 0, nnz);
            System.arraycopy(tempVals, 0, entryValues[row], 0, nnz);
        }
    }

    /** Deserialisation constructor. */
    public JLProjection(StreamInput in) throws IOException {
        this.inputDim = in.readVInt();
        this.outputDim = in.readVInt();
        this.colIndices = new int[outputDim][];
        this.entryValues = new float[outputDim][];
        for (int row = 0; row < outputDim; row++) {
            int nnz = in.readVInt();
            colIndices[row] = new int[nnz];
            entryValues[row] = new float[nnz];
            for (int i = 0; i < nnz; i++) {
                colIndices[row][i] = in.readVInt();
                entryValues[row][i] = in.readFloat();
            }
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(inputDim);
        out.writeVInt(outputDim);
        for (int row = 0; row < outputDim; row++) {
            int nnz = colIndices[row].length;
            out.writeVInt(nnz);
            for (int i = 0; i < nnz; i++) {
                out.writeVInt(colIndices[row][i]);
                out.writeFloat(entryValues[row][i]);
            }
        }
    }

    /**
     * Projects a single embedding vector from {@code inputDim} to {@code outputDim}.
     *
     * @param vector input embedding of length {@code inputDim}
     * @return projected vector of length {@code outputDim}
     * @throws IllegalArgumentException if {@code vector.length != inputDim}
     */
    public float[] project(float[] vector) {
        if (vector.length != inputDim) {
            throw new IllegalArgumentException("Expected vector of dimension [" + inputDim + "] but got [" + vector.length + "]");
        }
        float[] result = new float[outputDim];
        for (int row = 0; row < outputDim; row++) {
            float dot = 0;
            int[] cols = colIndices[row];
            float[] vals = entryValues[row];
            for (int i = 0; i < cols.length; i++) {
                dot += vals[i] * vector[cols[i]];
            }
            result[row] = dot;
        }
        return result;
    }

    /**
     * Projects a batch of embedding vectors.
     *
     * @param vectors array of embedding vectors, each of length {@code inputDim}
     * @return projected array, each element of length {@code outputDim}
     */
    public float[][] projectBatch(float[][] vectors) {
        float[][] results = new float[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            results[i] = project(vectors[i]);
        }
        return results;
    }

    public int getInputDim() {
        return inputDim;
    }

    public int getOutputDim() {
        return outputDim;
    }

    /**
     * Computes the recommended output dimension for a given number of points and
     * desired JL distortion {@code epsilon} using the bound
     * {@code d' = ceil(4 * log(n) / (epsilon^2 / 2 - epsilon^3 / 3))}.
     *
     * @param numPoints number of points to preserve pairwise distances for
     * @param epsilon   allowed relative distortion in (0, 1)
     * @return recommended {@code outputDim}
     */
    public static int recommendedOutputDim(int numPoints, double epsilon) {
        if (epsilon <= 0 || epsilon >= 1) {
            throw new IllegalArgumentException("epsilon must be in (0, 1), got [" + epsilon + "]");
        }
        double denom = epsilon * epsilon / 2.0 - epsilon * epsilon * epsilon / 3.0;
        return (int) Math.ceil(4.0 * Math.log(numPoints) / denom);
    }
}
