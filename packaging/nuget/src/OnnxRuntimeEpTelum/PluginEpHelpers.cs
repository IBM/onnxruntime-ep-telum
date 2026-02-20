using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace OnnxRuntimeEpTelum
{
    /// <summary>
    /// Helper APIs for locating and registering the Telum plugin EP from NuGet-based hosts.
    /// </summary>
    public static class PluginEpHelpers
    {
        private const string EpName = "TelumPluginExecutionProvider";
        private const string LibraryPathEnvVar = "ORT_TELUM_PLUGIN_EP_LIBRARY_PATH";

        public static string GetEpName()
        {
            return EpName;
        }

        public static IReadOnlyList<string> GetEpNames()
        {
            return new[] { EpName };
        }

        public static string GetLibraryPath()
        {
            var envPath = Environment.GetEnvironmentVariable(LibraryPathEnvVar);
            if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            {
                return envPath;
            }

            var fileName = GetPlatformLibraryFileName();
            var candidates = BuildCandidatePaths(fileName);

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            throw new FileNotFoundException(
                "Telum plugin EP shared library not found. Set " + LibraryPathEnvVar +
                " or place the library in one of these locations:" + Environment.NewLine +
                string.Join(Environment.NewLine, candidates));
        }

        private static List<string> BuildCandidatePaths(string fileName)
        {
            var candidates = new List<string>();

            var appBase = AppContext.BaseDirectory;
            var asmBase = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            var cwd = Directory.GetCurrentDirectory();

            var bases = new List<string>();
            if (!string.IsNullOrWhiteSpace(appBase)) bases.Add(appBase);
            if (!string.IsNullOrWhiteSpace(asmBase)) bases.Add(asmBase);
            if (!string.IsNullOrWhiteSpace(cwd)) bases.Add(cwd);

            foreach (var baseDir in bases)
            {
                candidates.Add(Path.Combine(baseDir, fileName));
                candidates.Add(Path.Combine(baseDir, "plugins", fileName));
                candidates.Add(Path.Combine(baseDir, "runtimes", "linux-s390x", "native", fileName));
                candidates.Add(Path.Combine(baseDir, "runtimes", "linux-s390", "native", fileName));
            }

            return candidates;
        }

        private static string GetPlatformLibraryFileName()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return "telum_plugin_ep.dll";
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return "libtelum_plugin_ep.dylib";
            }

            return "libtelum_plugin_ep.so";
        }
    }
}
