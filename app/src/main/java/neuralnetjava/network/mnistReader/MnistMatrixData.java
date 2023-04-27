package neuralnetjava.network.mnistReader;

import java.math.BigDecimal;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.javatuples.Pair;

import neuralnetjava.network.NeuralNetMatrixToolkit;

public class MnistMatrixData {

    private double [][] data;

    private double [] inputData;

    private Pair<RealMatrix, Integer> input;

    private int nRows;
    private int nCols;

    private int label;

    public MnistMatrixData(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        data = new double[nRows][nCols];
    }

    public double getValue(int r, int c) {
        return data[r][c];
    }

    public void setValue(int row, int col, int value) {
        data[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

    public void transformToInputData() {
        double inputDataArray[] = new double[nCols * nRows];
        int cnt = 0;
        
        for (int i = 0; i < nCols; i++) {
            for (int j = 0; j < nRows; j++) {
                inputDataArray[cnt] = data[i][j];
                cnt++;
            }
        }

        inputData = inputDataArray;
    }

    public void convertInputToRealMatrix() {
        this.transformToInputData();

        RealMatrix inputDMatrix = new Array2DRowRealMatrix(inputData);

        input = new Pair<RealMatrix, Integer>(NeuralNetMatrixToolkit.scalarMultiplication(inputDMatrix, 1.0/255.0), label);
    }

    public Pair<RealMatrix, Integer> getInput() {
        return input;
    }
}
