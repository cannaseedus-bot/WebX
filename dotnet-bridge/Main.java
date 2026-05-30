package micronaut.bridge;

import org.json.JSONObject;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.out.println("Usage: java -cp . micronaut.bridge.Main <op.json>");
            return;
        }
        String content = Files.readString(Paths.get(args[0]));
        JSONObject op = new JSONObject(content);
        JSONObject res = Dispatcher.dispatch(op);
        System.out.println(res.toString(2));
    }
}
