package micronaut.bridge;

import org.json.JSONObject;

public class Dispatcher {

    public static JSONObject dispatch(JSONObject op) {
        String code = op.optString("@op", "");
        switch (code) {
            case "DOTNET_RUN":
            case "DOTNET_MATH_ADD":
            case "DOTNET_SIMD_DOT":
            case "DOTNET_TENSOR_MATMUL":
            case "DOTNET_GPU_INFO":
                return DotNetBridge.run(op);
            default:
                JSONObject r = new JSONObject();
                r.put("error", "Unknown or unsupported opcode: " + code);
                return r;
        }
    }
}
