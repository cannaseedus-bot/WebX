package micronaut.bridge;

import org.json.JSONObject;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class DotNetBridge {

    public static JSONObject run(org.json.JSONObject op) {
        try {
            String payload = op.toString();
            // Path to built worker DLL (override via env for Release builds / different TFMs)
            String dllPath = System.getenv().getOrDefault(
                "MICRONAUT_DOTNET_WORKER_DLL",
                "dotnet-workers/Micronaut.Worker.Host/bin/Debug/net9.0/Micronaut.Worker.Host.dll"
            );
            ProcessBuilder pb = new ProcessBuilder("dotnet", dllPath, payload);
            pb.redirectErrorStream(true);
            Process proc = pb.start();
            InputStream is = proc.getInputStream();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int r;
            while ((r = is.read(buf)) != -1) baos.write(buf, 0, r);
            proc.waitFor();
            String out = new String(baos.toByteArray(), StandardCharsets.UTF_8).trim();
            if (out.isEmpty()) return new JSONObject().put("error", "no output");
            return new JSONObject(out);
        } catch (Exception e) {
            e.printStackTrace();
            return new JSONObject().put("error", e.toString());
        }
    }
}
