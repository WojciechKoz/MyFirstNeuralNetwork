import java.util.Random;

public class Main {
    private static float [][] inputs = {{1,0,0}, {1,1,0}, {0,1,0}, {0,0,1}};
    private static float [][] Doutput = {{0,1}, {0,1}, {0.7f, 0.3f}, {1,0}};

    private static NN network = new NN(2, new int[]{3, 2});

    private static void learning() {
        Random gen = new Random();

        for(int i = 0; i < 10000; i++) {
            System.out.print("proba numer " + i);

            int rand = gen.nextInt(4);
            System.out.println(" losowana liczba : " + rand);

            network.evolve(inputs[rand], Doutput[rand]);
        }
    }

    public static void main(String [] args) {
        learning();

        System.out.println(network.go(new float[]{1f, 0f, 0f}));
    }
}
