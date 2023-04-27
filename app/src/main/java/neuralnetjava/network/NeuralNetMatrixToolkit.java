package neuralnetjava.network;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;

import org.javatuples.Pair;

// dot, addition, multiplication, scalar multiplication, transpose

public class NeuralNetMatrixToolkit {
    public static RealMatrix add(RealMatrix a, RealMatrix b) {
        return a.add(b);
    }

    public static RealMatrix subtract(RealMatrix a, RealMatrix b) {
        return a.subtract(b);
    }

    public static RealMatrix multiply(RealMatrix a, RealMatrix b) {
        return a.multiply(b);
    }

    public static Pair<Integer, Integer> shape(RealMatrix a) {
        return new Pair<Integer,Integer>(a.getRowDimension(), a.getColumnDimension());
    }

    public static RealMatrix transpose(RealMatrix a) {
        return a.transpose();
    }

    public static RealMatrix exp(RealMatrix matrix) {
        int nRows = matrix.getRowDimension();
        int nCols = matrix.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(nRows, nCols);

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                result.setEntry(i, j, java.lang.Math.exp(matrix.getEntry(i, j)));
            }
        }

        return result;
    }

    public static RealMatrix scalarAddition(RealMatrix matrix, double scalar) {
        return matrix.scalarAdd(scalar);
    }

    public static RealMatrix scalarMultiplication(RealMatrix matrix, double scalar) {
        return matrix.scalarMultiply(scalar);
    }

    public static RealMatrix createUniformMatrix(double scalar, int nRows, int nCols) {
        RealMatrix result = MatrixUtils.createRealMatrix(nRows, nCols);

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                result.setEntry(i, j, scalar);
            }
        }

        return result;
    }

    public static RealMatrix createUniformMatrix(double scalar, Pair<Integer, Integer> shape) {
        RealMatrix result = MatrixUtils.createRealMatrix(shape.getValue0(), shape.getValue1());

        for (int i = 0; i < shape.getValue0(); i++) {
            for (int j = 0; j < shape.getValue1(); j++) {
                result.setEntry(i, j, scalar);
            }
        }

        return result;
    }

    public static RealMatrix power(RealMatrix matrix, int pow) {
        int nRows = matrix.getRowDimension();
        int nCols = matrix.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(nRows, nCols);

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                result.setEntry(i, j, java.lang.Math.pow(matrix.getEntry(i, j), pow));
            }
        }

        return result;
    }

    public static void setMatrix(ArrayList<RealMatrix> mxList, RealMatrix b, int index) {
        assert mxList.get(index).getRowDimension() == b.getRowDimension();
        assert mxList.get(index).getColumnDimension() == b.getColumnDimension();

        int nRows = mxList.get(index).getRowDimension();
        int nCols = mxList.get(index).getColumnDimension();

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                mxList.get(index).setEntry(i, j, b.getEntry(i, j));
            }
        }
    }

    public static RealMatrix HadamardProduct(RealMatrix matrix1, RealMatrix matrix2) {
        assert matrix1.getRowDimension() == matrix2.getRowDimension();
        assert matrix1.getColumnDimension() == matrix2.getColumnDimension();

        int nRows = matrix1.getRowDimension();
        int nCols = matrix1.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(nRows, nCols);

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                result.setEntry(i, j, matrix1.getEntry(i, j) * matrix2.getEntry(i, j));
            }
        }

        return result;
    }

    public static RealMatrix sigmoid(RealMatrix z) {
        return NeuralNetMatrixToolkit.power(
            NeuralNetMatrixToolkit.scalarAddition(
                NeuralNetMatrixToolkit.exp(
                    NeuralNetMatrixToolkit.scalarMultiplication(z, -1.0)), 1.0), -1);
    }

    public static RealMatrix sigmoid_derivate(RealMatrix z) {
        RealMatrix sigmoidMatrix = NeuralNetMatrixToolkit.sigmoid(z);

        return NeuralNetMatrixToolkit.HadamardProduct(sigmoidMatrix, NeuralNetMatrixToolkit.subtract(NeuralNetMatrixToolkit.createUniformMatrix(1.0, NeuralNetMatrixToolkit.shape(z)), sigmoidMatrix));
    }
}
