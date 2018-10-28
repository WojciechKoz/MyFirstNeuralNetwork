import java.util.Random;

public class Main {
    private static float [][] inputs = {{1,0,0}, {1,1,0}, {0,1,0}, {0,0,1}};
    private static float [][] Doutput = {{0,1}, {0,1}, {0.7f, 0.3f}, {1,0}};

    private static NN network = new NN(2, new int[]{3, 2});

    private static void learning() {
        Random gen = new Random();

        for(long i = 0; i < 100000; i++) {
            System.out.print("proba numer " + i);

            int rand = gen.nextInt(4);
            System.out.println(" losowana liczba : " + rand);

            network.evolve(inputs[rand], Doutput[rand]);
        }
    }

    public static void main(String [] args) {
        learning();

        System.out.println("010" + network.go(new float[]{0f, 1f, 0f}));
        System.out.println("110" + network.go(new float[]{1f, 1f, 0f}));
        System.out.println("100" + network.go(new float[]{1f, 0f, 0f}));
        System.out.println("001" + network.go(new float[]{0f, 0f, 1f}));
    }

    static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        long factor = (long) Math.pow(10, places);
        value = value * factor;
        long tmp = Math.round(value);
        return (double) tmp / factor;
    }
}
