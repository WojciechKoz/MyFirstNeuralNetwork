import java.util.ArrayList;
import static java.lang.Math.abs;

class NN {
    private ArrayList<ArrayList<Neuron>> neurons = new ArrayList<>();

    NN(int numOfL, int [] elementsInL) { // for example (2, [3, 2])
        for(int i = 0; i < numOfL; i++) {
            neurons.add(new ArrayList<>());

            for(int j = 0; j < elementsInL[i]; j++) {
                neurons.get(i).add(new Neuron(i+1 == numOfL ? 0 : elementsInL[i+1]));
            }
        }
    }

    void evolve(float[] input, float[] desiredOutput) {
        run(input);

        float learnRate = 0.02f;

        for(int j = 0; j < neurons.get(neurons.size()-1).size(); j++) {

            for(int k = 0; k < neurons.get(neurons.size()-2).size(); k++) {
                float localCost = Math.abs(neurons.get(neurons.size()-1).get(j).getValue() - desiredOutput[j]);

                neurons.get(neurons.size()-2).get(k).changeWeight(j, learnRate);

                run(input);

                float newCost = Math.abs(neurons.get(neurons.size()-1).get(j).getValue() - desiredOutput[j]);

                if(newCost < localCost) {
                    continue;
                }

                neurons.get(neurons.size()-2).get(k).changeWeight(j, -2* learnRate);


                run(input);

                newCost = Math.abs(neurons.get(neurons.size()-1).get(j).getValue() - desiredOutput[j]);

                if(newCost < localCost) {
                    continue;
                }

                neurons.get(neurons.size()-2).get(k).changeWeight(j, learnRate);
            }
        }
    }

    private void run(float[] input) {
        clear();

        for (int i = 0; i < neurons.get(0).size(); i++) {
            neurons.get(0).get(i).setValue(input[i]);
        }

        for(int i = 0; i < neurons.size()-1; i++) {
            for(int j = 0; j < neurons.get(i).size(); j++) {
                for(int k = 0; k < neurons.get(i+1).size(); k++) {
                   neurons.get(i+1).get(k).setValue(
                            neurons.get(i+1).get(k).getValue() +
                                    (neurons.get(i).get(j).getValue() * neurons.get(i).get(j).getWeights().get(k))
                    );
                }
            }

            for(int j = 0; j < neurons.get(i+1).size(); j++) {
                neurons.get(i+1).get(j).setValue(sigmoid(neurons.get(i+1).get(j).getValue()));
            }
        }
    }

    private void clear() {
        for(ArrayList<Neuron> layer : neurons) {
            for(Neuron neuron : layer) {
                neuron.setValue(0);
            }
        }
    }

    ArrayList<Float> go(float[] input) {
        run(input);

        ArrayList<Float> output = new ArrayList<>();

        for(Neuron neuron : neurons.get(neurons.size()-1)) {
            output.add(neuron.getValue());
        }

        return output;
    }



    float sigmoid(float x) {
        return (float) Main.round((4*x)/(1+ abs(4*x)), 5);
    }
}

class Neuron {
    private float value;
    private ArrayList<Float> weights = new ArrayList<>();
    private float bias;

    Neuron(int numOfW) {
        for(int i = 0; i < numOfW; i++) {
            weights.add(0.5f);
        }
        bias = 0;
    }

    ArrayList<Float> getWeights() {
        return weights;
    }

    void put(float v) {
        value = v;
    }

    float getBias() {
        return bias;
    }

    float getValue() {
        return value;
    }

    void setValue(float value) {
        this.value = value;
    }


    void changeWeight(int j, float learnRate) {
        weights.set(j, (float) Main.round(weights.get(j)+learnRate, 5));
    }
}
