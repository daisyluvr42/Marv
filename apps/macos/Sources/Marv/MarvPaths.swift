import Foundation

enum MarvEnv {
    static func path(_ key: String) -> String? {
        // Normalize env overrides once so UI + file IO stay consistent.
        guard let raw = getenv(key) else { return nil }
        let value = String(cString: raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty
        else {
            return nil
        }
        return value
    }
}

enum MarvPaths {
    private static let configPathEnv = ["MARV_CONFIG_PATH"]
    private static let stateDirEnv = ["MARV_STATE_DIR"]
    private static let agentDirEnv = ["MARV_AGENT_DIR", "PI_CODING_AGENT_DIR"]

    static var stateDirURL: URL {
        for key in self.stateDirEnv {
            if let override = MarvEnv.path(key) {
                return URL(fileURLWithPath: override, isDirectory: true)
            }
        }
        let home = FileManager().homeDirectoryForCurrentUser
        return home.appendingPathComponent(".marv", isDirectory: true)
    }

    private static func resolveConfigCandidate(in dir: URL) -> URL? {
        let candidates = [
            dir.appendingPathComponent("marv.json"),
        ]
        return candidates.first(where: { FileManager().fileExists(atPath: $0.path) })
    }

    static var configURL: URL {
        for key in self.configPathEnv {
            if let override = MarvEnv.path(key) {
                return URL(fileURLWithPath: override)
            }
        }
        let stateDir = self.stateDirURL
        if let existing = self.resolveConfigCandidate(in: stateDir) {
            return existing
        }
        return stateDir.appendingPathComponent("marv.json")
    }

    static var workspaceURL: URL {
        self.stateDirURL.appendingPathComponent("workspace", isDirectory: true)
    }

    static var mainAgentDirURL: URL {
        for key in self.agentDirEnv {
            if let override = MarvEnv.path(key) {
                return URL(fileURLWithPath: override, isDirectory: true)
            }
        }
        return self.stateDirURL
            .appendingPathComponent("agents", isDirectory: true)
            .appendingPathComponent("main", isDirectory: true)
            .appendingPathComponent("agent", isDirectory: true)
    }

    static var authProfilesURL: URL {
        self.mainAgentDirURL.appendingPathComponent("auth-profiles.json")
    }
}
