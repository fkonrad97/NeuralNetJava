package neuralnetjava.network;

import java.util.*;

import java.util.ArrayList;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.javatuples.Pair;
import com.google.common.collect.Lists;


/*
 * [2, 3, 1] neural net. 2 input parameter, 3 hidden layer and 1 output neuron.
 * The hidden 3 neuron got two weighted arrow each from the 2 input neuron and each 3 hidden neuron has a bias.
 * The output neuron has 3 weighted arrow from each hidden layer neuron and has of course a bias.
 * So the weights in each layer means the incoming links from the previous layer.
 * [
 *  [
 *      Neuron[bias=0.5669194035304935, weights=[0.7007636053254868, 1.728062541444258]], 
 *      Neuron[bias=0.14163003071942695, weights=[-0.543913923572973, 0.2225102759048273]], 
 *      Neuron[bias=0.35622278303832405, weights=[-0.6259445456052625, 0.8392591457256069]
 *  ]
 * ], 
 * [
 *      Neuron[bias=0.24357304514540604, weights=[-0.8661404003822548, 1.8552254709993192, -1.7390002625384982]
 * ]]]
 */

public class NeuralNet {
    private static final Random rand = new Random();
    private ArrayList<RealMatrix> weights = new ArrayList<>();
    private ArrayList<RealMatrix> biases = new ArrayList<>();
    private int[] layers = null;

    public NeuralNet(int[] layers) {
        this.layers = layers;

        for (int i = 1; i < layers.length; i++) {
            int neuronNum = layers[i];
            int prevLayerNeuronNum = layers[i - 1];

            double[] tmpBiases = new double[neuronNum];
            RealMatrix tmpWeights = new Array2DRowRealMatrix(neuronNum, prevLayerNeuronNum);

            for (int j = 0; j < neuronNum; j++) {
                tmpBiases[j] = rand.nextGaussian();
            
                for (int z = 0; z < prevLayerNeuronNum; z++) {
                    tmpWeights.setEntry(j, z, rand.nextGaussian());
                }
            }

            biases.add(new Array2DRowRealMatrix(tmpBiases));
            weights.add(tmpWeights);
        }
    }

    public int[] getLayers() {
        return layers;
    }

    /**
     * 
     * @param input - should be in {{x},{z}} form
     * @return
     */
    public RealMatrix feedforward(RealMatrix input) {
        RealMatrix result = input;
        for (int i = 0; i < layers.length - 1; i++) {
            result = NeuralNetMatrixToolkit.sigmoid(NeuralNetMatrixToolkit.add(NeuralNetMatrixToolkit.multiply(weights.get(i), result), biases.get(i)));
        }

        return result;
    }

    public void SGD(ArrayList<Pair<RealMatrix, Integer>> training_data, int epoch, int mini_batch_size, double eta) {
        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(training_data);

            List<List<Pair<RealMatrix, Integer>>> mini_batches = Lists.partition(training_data, mini_batch_size);
            
            for (List<Pair<RealMatrix, Integer>> mini_batch_list : mini_batches) {
                // ArrayList<Pair<RealMatrix, Integer>> mini_batch = new ArrayList<>(mini_batch_list);
                this.update_mini_batch(mini_batch_list, eta);
            }

            System.out.println("Epoch " + i + " complete!");
        }
    }

    public void SGD(ArrayList<Pair<RealMatrix, Integer>> training_data, int epoch, int mini_batch_size, double eta, ArrayList<Pair<RealMatrix, Integer>> test_data) {
        int n_test = test_data.size();

        for (int i = 0; i < epoch; i++) {
            Collections.shuffle(training_data);

            List<List<Pair<RealMatrix, Integer>>> mini_batches = Lists.partition(training_data, mini_batch_size);
            
            for (List<Pair<RealMatrix, Integer>> mini_batch_list : mini_batches) {
                this.update_mini_batch(mini_batch_list, eta);
            }
            System.out.println(this.biases.get(1));

            System.out.println("Epoch " + i + ": " + this.mnist_evaluate(test_data) + " / " + n_test);
        }
    }

    public void update_mini_batch(List<Pair<RealMatrix, Integer>> mini_batch, double eta) {
        ArrayList<RealMatrix> nabla_b = new ArrayList<>();
        ArrayList<RealMatrix> nabla_w = new ArrayList<>();
        
        for (int i = 0; i < layers.length - 1; i++) {
            nabla_b.add(NeuralNetMatrixToolkit.createUniformMatrix(0.0, NeuralNetMatrixToolkit.shape(biases.get(i))));
            nabla_w.add(NeuralNetMatrixToolkit.createUniformMatrix(0.0, NeuralNetMatrixToolkit.shape(weights.get(i))));
        }

        for (Pair<RealMatrix, Integer> element : mini_batch) {
            RealMatrix input = element.getValue0();
            Integer y = element.getValue1();

            Pair<ArrayList<RealMatrix>, ArrayList<RealMatrix>> backpropPair = this.backpropagation(input, y);
            ArrayList<RealMatrix> delta_nabla_b = backpropPair.getValue0();
            ArrayList<RealMatrix> delta_nabla_w = backpropPair.getValue1();

            for (int i = 0; i < nabla_b.size(); i++) {
                NeuralNetMatrixToolkit.setMatrix(nabla_b, NeuralNetMatrixToolkit.add(delta_nabla_b.get(i), nabla_b.get(i)), i);
                NeuralNetMatrixToolkit.setMatrix(nabla_w, NeuralNetMatrixToolkit.add(delta_nabla_w.get(i), nabla_w.get(i)), i);
            }
        }

        for (int i = 0; i < this.weights.size(); i++) {
            NeuralNetMatrixToolkit.setMatrix(this.weights, NeuralNetMatrixToolkit.subtract(this.weights.get(i), NeuralNetMatrixToolkit.scalarMultiplication(nabla_w.get(i), eta / mini_batch.size())), i);
            NeuralNetMatrixToolkit.setMatrix(this.biases, NeuralNetMatrixToolkit.subtract(this.biases.get(i), NeuralNetMatrixToolkit.scalarMultiplication(nabla_b.get(i), eta / mini_batch.size())), i);
        }
    }

    public Pair<ArrayList<RealMatrix>, ArrayList<RealMatrix>> backpropagation(RealMatrix input, Integer y) {
        ArrayList<RealMatrix> nabla_b = new ArrayList<>();
        ArrayList<RealMatrix> nabla_w = new ArrayList<>();
        ArrayList<RealMatrix> activations = new ArrayList<>();
        ArrayList<RealMatrix> zs = new ArrayList<>();
        
        for (int i = 0; i < layers.length - 1; i++) {
            nabla_b.add(NeuralNetMatrixToolkit.createUniformMatrix(0.0, NeuralNetMatrixToolkit.shape(biases.get(i))));
            nabla_w.add(NeuralNetMatrixToolkit.createUniformMatrix(0.0, NeuralNetMatrixToolkit.shape(weights.get(i))));
        }

        // Feedforward
        RealMatrix activation = input;
        activations.add(activation);

        for (int i = 0; i < layers.length - 1; i++) {
            RealMatrix z = NeuralNetMatrixToolkit.add(NeuralNetMatrixToolkit.multiply(weights.get(i), activation), biases.get(i));
            zs.add(z);
            activation = NeuralNetMatrixToolkit.sigmoid(z);
            activations.add(activation);
        }

        // activation size = layers.length
        // zs size = layers.length - 1
        // weights / biases size = layers.length - 1

        // Backward pass
        RealMatrix delta = NeuralNetMatrixToolkit.HadamardProduct(
            this.cost_derivative(activations.get(activations.size() - 1), y), 
            NeuralNetMatrixToolkit.sigmoid_derivate(zs.get(zs.size() - 1)));

        NeuralNetMatrixToolkit.setMatrix(nabla_b, delta, nabla_b.size() - 1);
        NeuralNetMatrixToolkit.setMatrix(nabla_w, NeuralNetMatrixToolkit.multiply(delta, NeuralNetMatrixToolkit.transpose(activations.get(activations.size() - 2))), nabla_w.size() - 1);

        // index starts denote from the second last layer
        for (int i = layers.length - 2; i > 0; i--) { 
            RealMatrix sp = NeuralNetMatrixToolkit.sigmoid_derivate(zs.get(i - 1));

            delta = NeuralNetMatrixToolkit.HadamardProduct(NeuralNetMatrixToolkit.multiply(NeuralNetMatrixToolkit.transpose(weights.get(i)) ,delta), sp);

            NeuralNetMatrixToolkit.setMatrix(nabla_b, delta, i - 1);
            NeuralNetMatrixToolkit.setMatrix(nabla_w, NeuralNetMatrixToolkit.multiply(delta, NeuralNetMatrixToolkit.transpose(activations.get(i - 1))), i - 1);
        }

        return new Pair<>(nabla_b, nabla_w);
    }

    public int mnist_evaluate(ArrayList<Pair<RealMatrix, Integer>> test_data) {
        int test_result = 0;
        for (Pair<RealMatrix, Integer> dataPair : test_data) {
            double[] res = this.feedforward(dataPair.getValue0()).getColumn(0);
            ArrayList<Double> list = new ArrayList<Double>();
            for (double x : res) {
                list.add(x);
            }

            int index = list.indexOf(Collections.max(list));
            int label = dataPair.getValue1(); 

            if (index == label) test_result++;
        }
        return test_result;
    }

    public RealMatrix cost_derivative(RealMatrix output_activations, Integer y) {
        return NeuralNetMatrixToolkit.subtract(output_activations, NeuralNetMatrixToolkit.createUniformMatrix((double) y, NeuralNetMatrixToolkit.shape(output_activations)));
    }

    @Override
    public String toString() {
        return "NeuralNet [weights=" + weights + ", biases=" + biases + "]";
    }
}