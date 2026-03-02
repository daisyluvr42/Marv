import Foundation
import Testing
@testable import Marv

@Suite(.serialized)
struct MarvConfigFileTests {
    @Test
    func configPathRespectsEnvOverride() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("marv-config-\(UUID().uuidString)")
            .appendingPathComponent("marv.json")
            .path

        await TestIsolation.withEnvValues(["MARV_CONFIG_PATH": override]) {
            #expect(MarvConfigFile.url().path == override)
        }
    }

    @MainActor
    @Test
    func remoteGatewayPortParsesAndMatchesHost() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("marv-config-\(UUID().uuidString)")
            .appendingPathComponent("marv.json")
            .path

        await TestIsolation.withEnvValues(["MARV_CONFIG_PATH": override]) {
            MarvConfigFile.saveDict([
                "gateway": [
                    "remote": [
                        "url": "ws://gateway.ts.net:19999",
                    ],
                ],
            ])
            #expect(MarvConfigFile.remoteGatewayPort() == 19999)
            #expect(MarvConfigFile.remoteGatewayPort(matchingHost: "gateway.ts.net") == 19999)
            #expect(MarvConfigFile.remoteGatewayPort(matchingHost: "gateway") == 19999)
            #expect(MarvConfigFile.remoteGatewayPort(matchingHost: "other.ts.net") == nil)
        }
    }

    @MainActor
    @Test
    func setRemoteGatewayUrlPreservesScheme() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("marv-config-\(UUID().uuidString)")
            .appendingPathComponent("marv.json")
            .path

        await TestIsolation.withEnvValues(["MARV_CONFIG_PATH": override]) {
            MarvConfigFile.saveDict([
                "gateway": [
                    "remote": [
                        "url": "wss://old-host:111",
                    ],
                ],
            ])
            MarvConfigFile.setRemoteGatewayUrl(host: "new-host", port: 2222)
            let root = MarvConfigFile.loadDict()
            let url = ((root["gateway"] as? [String: Any])?["remote"] as? [String: Any])?["url"] as? String
            #expect(url == "wss://new-host:2222")
        }
    }

    @Test
    func stateDirOverrideSetsConfigPath() async {
        let dir = FileManager().temporaryDirectory
            .appendingPathComponent("marv-state-\(UUID().uuidString)", isDirectory: true)
            .path

        await TestIsolation.withEnvValues([
            "MARV_CONFIG_PATH": nil,
            "MARV_STATE_DIR": dir,
        ]) {
            #expect(MarvConfigFile.stateDirURL().path == dir)
            #expect(MarvConfigFile.url().path == "\(dir)/marv.json")
        }
    }

    @MainActor
    @Test
    func saveDictAppendsConfigAuditLog() async throws {
        let stateDir = FileManager().temporaryDirectory
            .appendingPathComponent("marv-state-\(UUID().uuidString)", isDirectory: true)
        let configPath = stateDir.appendingPathComponent("marv.json")
        let auditPath = stateDir.appendingPathComponent("logs/config-audit.jsonl")

        defer { try? FileManager().removeItem(at: stateDir) }

        try await TestIsolation.withEnvValues([
            "MARV_STATE_DIR": stateDir.path,
            "MARV_CONFIG_PATH": configPath.path,
        ]) {
            MarvConfigFile.saveDict([
                "gateway": ["mode": "local"],
            ])

            let configData = try Data(contentsOf: configPath)
            let configRoot = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
            #expect((configRoot?["meta"] as? [String: Any]) != nil)

            let rawAudit = try String(contentsOf: auditPath, encoding: .utf8)
            let lines = rawAudit
                .split(whereSeparator: \.isNewline)
                .map(String.init)
            #expect(!lines.isEmpty)
            guard let last = lines.last else {
                Issue.record("Missing config audit line")
                return
            }
            let auditRoot = try JSONSerialization.jsonObject(with: Data(last.utf8)) as? [String: Any]
            #expect(auditRoot?["source"] as? String == "macos-marv-config-file")
            #expect(auditRoot?["event"] as? String == "config.write")
            #expect(auditRoot?["result"] as? String == "success")
            #expect(auditRoot?["configPath"] as? String == configPath.path)
        }
    }
}
